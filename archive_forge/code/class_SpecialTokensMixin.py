import copy
import json
import os
import re
import warnings
from collections import UserDict
from collections.abc import Mapping, Sized
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import numpy as np
from packaging import version
from . import __version__
from .dynamic_module_utils import custom_object_save
from .utils import (
class SpecialTokensMixin:
    """
    A mixin derived by [`PreTrainedTokenizer`] and [`PreTrainedTokenizerFast`] to handle specific behaviors related to
    special tokens. In particular, this class hold the attributes which can be used to directly access these special
    tokens in a model-independent manner and allow to set and update the special tokens.

    Args:
        bos_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing the beginning of a sentence.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing the end of a sentence.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing an out-of-vocabulary token.
        sep_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token separating two different sentences in the same input (used by BERT for instance).
        pad_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.
        cls_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing the class of the input (used by BERT for instance).
        mask_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing a masked token (used by masked-language modeling pretraining objectives, like
            BERT).
        additional_special_tokens (tuple or list of `str` or `tokenizers.AddedToken`, *optional*):
            A tuple or a list of additional tokens, which will be marked as `special`, meaning that they will be
            skipped when decoding if `skip_special_tokens` is set to `True`.
    """
    SPECIAL_TOKENS_ATTRIBUTES = ['bos_token', 'eos_token', 'unk_token', 'sep_token', 'pad_token', 'cls_token', 'mask_token', 'additional_special_tokens']

    def __init__(self, verbose=False, **kwargs):
        self._bos_token = None
        self._eos_token = None
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._pad_token_type_id = 0
        self._additional_special_tokens = []
        self.verbose = verbose
        for key, value in kwargs.items():
            if value is None:
                continue
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if key == 'additional_special_tokens':
                    assert isinstance(value, (list, tuple)), f'Value {value} is not a list or tuple'
                    assert all((isinstance(t, (str, AddedToken)) for t in value)), 'One of the tokens is not a string or an AddedToken'
                    setattr(self, key, value)
                elif isinstance(value, (str, AddedToken)):
                    setattr(self, key, value)
                else:
                    raise TypeError(f'Special token {key} has to be either str or AddedToken but got: {type(value)}')

    def sanitize_special_tokens(self) -> int:
        """
        The `sanitize_special_tokens` is now deprecated kept for backward compatibility and will be removed in
        transformers v5.
        """
        logger.warning_once('The `sanitize_special_tokens` will be removed in transformers v5.')
        return self.add_tokens(self.all_special_tokens_extended, special_tokens=True)

    def add_special_tokens(self, special_tokens_dict: Dict[str, Union[str, AddedToken]], replace_additional_special_tokens=True) -> int:
        """
        Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If
        special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the
        current vocabulary).

        When adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix of the
        model so that its embedding matrix matches the tokenizer.

        In order to do that, please use the [`~PreTrainedModel.resize_token_embeddings`] method.

        Using `add_special_tokens` will ensure your special tokens can be used in several ways:

        - Special tokens can be skipped when decoding using `skip_special_tokens = True`.
        - Special tokens are carefully handled by the tokenizer (they are never split), similar to `AddedTokens`.
        - You can easily refer to special tokens using tokenizer class attributes like `tokenizer.cls_token`. This
          makes it easy to develop model-agnostic training and fine-tuning scripts.

        When possible, special tokens are already registered for provided pretrained models (for instance
        [`BertTokenizer`] `cls_token` is already registered to be :obj*'[CLS]'* and XLM's one is also registered to be
        `'</s>'`).

        Args:
            special_tokens_dict (dictionary *str* to *str* or `tokenizers.AddedToken`):
                Keys should be in the list of predefined special attributes: [`bos_token`, `eos_token`, `unk_token`,
                `sep_token`, `pad_token`, `cls_token`, `mask_token`, `additional_special_tokens`].

                Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer
                assign the index of the `unk_token` to them).
            replace_additional_special_tokens (`bool`, *optional*,, defaults to `True`):
                If `True`, the existing list of additional special tokens will be replaced by the list provided in
                `special_tokens_dict`. Otherwise, `self._additional_special_tokens` is just extended. In the former
                case, the tokens will NOT be removed from the tokenizer's full vocabulary - they are only being flagged
                as non-special tokens. Remember, this only affects which tokens are skipped during decoding, not the
                `added_tokens_encoder` and `added_tokens_decoder`. This means that the previous
                `additional_special_tokens` are still added tokens, and will not be split by the model.

        Returns:
            `int`: Number of tokens added to the vocabulary.

        Examples:

        ```python
        # Let's see how to add a new classification token to GPT-2
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        model = GPT2Model.from_pretrained("openai-community/gpt2")

        special_tokens_dict = {"cls_token": "<CLS>"}

        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print("We have added", num_added_toks, "tokens")
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))

        assert tokenizer.cls_token == "<CLS>"
        ```"""
        if not special_tokens_dict:
            return 0
        added_tokens = []
        for key, value in special_tokens_dict.items():
            assert key in self.SPECIAL_TOKENS_ATTRIBUTES, f'Key {key} is not a special token'
            if self.verbose:
                logger.info(f'Assigning {value} to the {key} key of the tokenizer')
            if key == 'additional_special_tokens':
                assert isinstance(value, (list, tuple)) and all((isinstance(t, (str, AddedToken)) for t in value)), f'Tokens {value} for key {key} should all be str or AddedToken instances'
                to_add = []
                for token in value:
                    if isinstance(token, str):
                        token = AddedToken(token, rstrip=False, lstrip=False, normalized=False, special=True)
                    if not replace_additional_special_tokens and str(token) in self.additional_special_tokens:
                        continue
                    to_add.append(token)
                if replace_additional_special_tokens and len(to_add) > 0:
                    setattr(self, key, list(to_add))
                else:
                    self._additional_special_tokens.extend(to_add)
                added_tokens += to_add
            else:
                if not isinstance(value, (str, AddedToken)):
                    raise ValueError(f'Token {value} for key {key} should be a str or an AddedToken instance')
                if isinstance(value, str):
                    value = AddedToken(value, rstrip=False, lstrip=False, normalized=False, special=True)
                if isinstance(value, AddedToken):
                    setattr(self, key, value)
                if value not in added_tokens:
                    added_tokens.append(value)
        added_tokens = self.add_tokens(added_tokens, special_tokens=True)
        return added_tokens

    def add_tokens(self, new_tokens: Union[str, AddedToken, List[Union[str, AddedToken]]], special_tokens: bool=False) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary and and will be isolated before the tokenization
        algorithm is applied. Added tokens and tokens from the vocabulary of the tokenization algorithm are therefore
        not treated in the same way.

        Note, when adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix
        of the model so that its embedding matrix matches the tokenizer.

        In order to do that, please use the [`~PreTrainedModel.resize_token_embeddings`] method.

        Args:
            new_tokens (`str`, `tokenizers.AddedToken` or a list of *str* or `tokenizers.AddedToken`):
                Tokens are only added if they are not already in the vocabulary. `tokenizers.AddedToken` wraps a string
                token to let you personalize its behavior: whether this token should only match against a single word,
                whether this token should strip all potential whitespaces on the left side, whether this token should
                strip all potential whitespaces on the right side, etc.
            special_tokens (`bool`, *optional*, defaults to `False`):
                Can be used to specify if the token is a special token. This mostly change the normalization behavior
                (special tokens like CLS or [MASK] are usually not lower-cased for instance).

                See details for `tokenizers.AddedToken` in HuggingFace tokenizers library.

        Returns:
            `int`: Number of tokens added to the vocabulary.

        Examples:

        ```python
        # Let's see how to increase the vocabulary of Bert model and tokenizer
        tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
        model = BertModel.from_pretrained("google-bert/bert-base-uncased")

        num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
        print("We have added", num_added_toks, "tokens")
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))
        ```"""
        if not new_tokens:
            return 0
        if not isinstance(new_tokens, (list, tuple)):
            new_tokens = [new_tokens]
        return self._add_tokens(new_tokens, special_tokens=special_tokens)

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool=False) -> int:
        raise NotImplementedError

    @property
    def bos_token(self) -> str:
        """
        `str`: Beginning of sentence token. Log an error if used while not having been set.
        """
        if self._bos_token is None:
            if self.verbose:
                logger.error('Using bos_token, but it is not set yet.')
            return None
        return str(self._bos_token)

    @property
    def eos_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._eos_token is None:
            if self.verbose:
                logger.error('Using eos_token, but it is not set yet.')
            return None
        return str(self._eos_token)

    @property
    def unk_token(self) -> str:
        """
        `str`: Unknown token. Log an error if used while not having been set.
        """
        if self._unk_token is None:
            if self.verbose:
                logger.error('Using unk_token, but it is not set yet.')
            return None
        return str(self._unk_token)

    @property
    def sep_token(self) -> str:
        """
        `str`: Separation token, to separate context and query in an input sequence. Log an error if used while not
        having been set.
        """
        if self._sep_token is None:
            if self.verbose:
                logger.error('Using sep_token, but it is not set yet.')
            return None
        return str(self._sep_token)

    @property
    def pad_token(self) -> str:
        """
        `str`: Padding token. Log an error if used while not having been set.
        """
        if self._pad_token is None:
            if self.verbose:
                logger.error('Using pad_token, but it is not set yet.')
            return None
        return str(self._pad_token)

    @property
    def cls_token(self) -> str:
        """
        `str`: Classification token, to extract a summary of an input sequence leveraging self-attention along the full
        depth of the model. Log an error if used while not having been set.
        """
        if self._cls_token is None:
            if self.verbose:
                logger.error('Using cls_token, but it is not set yet.')
            return None
        return str(self._cls_token)

    @property
    def mask_token(self) -> str:
        """
        `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
        having been set.
        """
        if self._mask_token is None:
            if self.verbose:
                logger.error('Using mask_token, but it is not set yet.')
            return None
        return str(self._mask_token)

    @property
    def additional_special_tokens(self) -> List[str]:
        """
        `List[str]`: All the additional special tokens you may want to use. Log an error if used while not having been
        set.
        """
        if self._additional_special_tokens is None:
            if self.verbose:
                logger.error('Using additional_special_tokens, but it is not set yet.')
            return None
        return [str(tok) for tok in self._additional_special_tokens]

    @bos_token.setter
    def bos_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError('Cannot set a non-string value as the BOS token')
        self._bos_token = value

    @eos_token.setter
    def eos_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError('Cannot set a non-string value as the EOS token')
        self._eos_token = value

    @unk_token.setter
    def unk_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError('Cannot set a non-string value as the UNK token')
        self._unk_token = value

    @sep_token.setter
    def sep_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError('Cannot set a non-string value as the SEP token')
        self._sep_token = value

    @pad_token.setter
    def pad_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError('Cannot set a non-string value as the PAD token')
        self._pad_token = value

    @cls_token.setter
    def cls_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError('Cannot set a non-string value as the CLS token')
        self._cls_token = value

    @mask_token.setter
    def mask_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError('Cannot set a non-string value as the MASK token')
        self._mask_token = value

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value if value is not None else None

    @property
    def bos_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the beginning of sentence token in the vocabulary. Returns `None` if the token has not
        been set.
        """
        if self._bos_token is None:
            return None
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the end of sentence token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        if self._eos_token is None:
            return None
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def unk_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the unknown token in the vocabulary. Returns `None` if the token has not been set.
        """
        if self._unk_token is None:
            return None
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def sep_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the separation token in the vocabulary, to separate context and query in an input
        sequence. Returns `None` if the token has not been set.
        """
        if self._sep_token is None:
            return None
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def pad_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the padding token in the vocabulary. Returns `None` if the token has not been set.
        """
        if self._pad_token is None:
            return None
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def pad_token_type_id(self) -> int:
        """
        `int`: Id of the padding token type in the vocabulary.
        """
        return self._pad_token_type_id

    @property
    def cls_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the classification token in the vocabulary, to extract a summary of an input sequence
        leveraging self-attention along the full depth of the model.

        Returns `None` if the token has not been set.
        """
        if self._cls_token is None:
            return None
        return self.convert_tokens_to_ids(self.cls_token)

    @property
    def mask_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the mask token in the vocabulary, used when training a model with masked-language
        modeling. Returns `None` if the token has not been set.
        """
        if self._mask_token is None:
            return None
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def additional_special_tokens_ids(self) -> List[int]:
        """
        `List[int]`: Ids of all the additional special tokens in the vocabulary. Log an error if used while not having
        been set.
        """
        return self.convert_tokens_to_ids(self.additional_special_tokens)

    @bos_token_id.setter
    def bos_token_id(self, value):
        self._bos_token = self.convert_ids_to_tokens(value) if value is not None else None

    @eos_token_id.setter
    def eos_token_id(self, value):
        self._eos_token = self.convert_ids_to_tokens(value) if value is not None else None

    @unk_token_id.setter
    def unk_token_id(self, value):
        self._unk_token = self.convert_ids_to_tokens(value) if value is not None else None

    @sep_token_id.setter
    def sep_token_id(self, value):
        self._sep_token = self.convert_ids_to_tokens(value) if value is not None else None

    @pad_token_id.setter
    def pad_token_id(self, value):
        self._pad_token = self.convert_ids_to_tokens(value) if value is not None else None

    @cls_token_id.setter
    def cls_token_id(self, value):
        self._cls_token = self.convert_ids_to_tokens(value) if value is not None else None

    @mask_token_id.setter
    def mask_token_id(self, value):
        self._mask_token = self.convert_ids_to_tokens(value) if value is not None else None

    @additional_special_tokens_ids.setter
    def additional_special_tokens_ids(self, values):
        self._additional_special_tokens = [self.convert_ids_to_tokens(value) for value in values]

    @property
    def special_tokens_map(self) -> Dict[str, Union[str, List[str]]]:
        """
        `Dict[str, Union[str, List[str]]]`: A dictionary mapping special token class attributes (`cls_token`,
        `unk_token`, etc.) to their values (`'<unk>'`, `'<cls>'`, etc.).

        Convert potential tokens of `tokenizers.AddedToken` type to string.
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def special_tokens_map_extended(self) -> Dict[str, Union[str, AddedToken, List[Union[str, AddedToken]]]]:
        """
        `Dict[str, Union[str, tokenizers.AddedToken, List[Union[str, tokenizers.AddedToken]]]]`: A dictionary mapping
        special token class attributes (`cls_token`, `unk_token`, etc.) to their values (`'<unk>'`, `'<cls>'`, etc.).

        Don't convert tokens of `tokenizers.AddedToken` type to string so they can be used to control more finely how
        special tokens are tokenized.
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, '_' + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens_extended(self) -> List[Union[str, AddedToken]]:
        """
        `List[Union[str, tokenizers.AddedToken]]`: All the special tokens (`'<unk>'`, `'<cls>'`, etc.), the order has
        nothing to do with the index of each tokens. If you want to know the correct indices, check
        `self.added_tokens_encoder`. We can't create an order anymore as the keys are `AddedTokens` and not `Strings`.

        Don't convert tokens of `tokenizers.AddedToken` type to string so they can be used to control more finely how
        special tokens are tokenized.
        """
        all_tokens = []
        seen = set()
        for value in self.special_tokens_map_extended.values():
            if isinstance(value, (list, tuple)):
                tokens_to_add = [token for token in value if str(token) not in seen]
            else:
                tokens_to_add = [value] if str(value) not in seen else []
            seen.update(map(str, tokens_to_add))
            all_tokens.extend(tokens_to_add)
        return all_tokens

    @property
    def all_special_tokens(self) -> List[str]:
        """
        `List[str]`: A list of the unique special tokens (`'<unk>'`, `'<cls>'`, ..., etc.).

        Convert tokens of `tokenizers.AddedToken` type to string.
        """
        all_toks = [str(s) for s in self.all_special_tokens_extended]
        return all_toks

    @property
    def all_special_ids(self) -> List[int]:
        """
        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids