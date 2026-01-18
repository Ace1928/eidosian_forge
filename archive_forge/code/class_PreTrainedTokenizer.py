import bisect
import itertools
import re
import unicodedata
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union, overload
from .tokenization_utils_base import (
from .utils import PaddingStrategy, TensorType, add_end_docstrings, logging
@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class PreTrainedTokenizer(PreTrainedTokenizerBase):
    """
    Base class for all slow tokenizers.

    Inherits from [`~tokenization_utils_base.PreTrainedTokenizerBase`].

    Handle all the shared methods for tokenization and special tokens as well as methods downloading/caching/loading
    pretrained tokenizers as well as adding tokens to the vocabulary.

    This class also contain the added tokens in a unified way on top of all tokenizers so we don't have to handle the
    specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).
    """

    def __init__(self, **kwargs):
        self.tokens_trie = Trie()
        if not hasattr(self, '_added_tokens_decoder'):
            self._added_tokens_decoder: Dict[int, AddedToken] = {}
        self._added_tokens_decoder.update(kwargs.pop('added_tokens_decoder', {}))
        self._added_tokens_encoder: Dict[str, int] = {k.content: v for v, k in self._added_tokens_decoder.items()}
        super().__init__(**kwargs)
        self._add_tokens([token for token in self.all_special_tokens_extended if token not in self._added_tokens_encoder], special_tokens=True)
        self._decode_use_source_tokenizer = False

    @property
    def is_fast(self) -> bool:
        return False

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        raise NotImplementedError

    @property
    def added_tokens_encoder(self) -> Dict[str, int]:
        """
        Returns the sorted mapping from string to index. The added tokens encoder is cached for performance
        optimisation in `self._added_tokens_encoder` for the slow tokenizers.
        """
        return {k.content: v for v, k in sorted(self._added_tokens_decoder.items(), key=lambda item: item[0])}

    @property
    def added_tokens_decoder(self) -> Dict[int, AddedToken]:
        """
        Returns the added tokens in the vocabulary as a dictionary of index to AddedToken.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        return dict(sorted(self._added_tokens_decoder.items(), key=lambda item: item[0]))

    @added_tokens_decoder.setter
    def added_tokens_decoder(self, value: Dict[int, Union[AddedToken, str]]) -> Dict[int, AddedToken]:
        for index, token in value.items():
            if not isinstance(token, (str, AddedToken)) or not isinstance(index, int):
                raise ValueError(f'The provided `added_tokens_decoder` has an element of type {(index.__class__, token.__class__)}, should be a dict of {(int, Union[AddedToken, str])}')
            self._added_tokens_decoder[index] = AddedToken(token) if isinstance(token, str) else token
            self._added_tokens_encoder[str(token)] = index

    def get_added_vocab(self) -> Dict[str, int]:
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index. Results might be different from
        the fast call because for now we always add the tokens even if they are already in the vocabulary. This is
        something we should change.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        return self._added_tokens_encoder

    def __len__(self):
        """
        Size of the full vocabulary with the added tokens. Counts the `keys` and not the `values` because otherwise if
        there is a hole in the vocab, we will add tokenizers at a wrong index.
        """
        return len(set(self.get_vocab().keys()))

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool=False) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary. Special tokens are sometimes already in the
        vocab which is why they have to be handled specifically.

        Args:
            new_tokens (`List[str]`or `List[tokenizers.AddedToken]`):
                Token(s) to add in vocabulary. A token is counted as added if it's not already in the vocabulary
                (tested by checking if the tokenizer assign the index of the `unk_token` to them). If a token is part
                of the vocabulary then we simply mark this token as an `AddedToken` which allows to control the
                stripping and normalization of this token. This is NOT possible in `tokenizers`.
            special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the tokens should be added as special tokens.

        Returns:
            `int`: The number of tokens actually added to the vocabulary.

        Examples:

        ```python
        # Let's see how to increase the vocabulary of Bert model and tokenizer
        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        model = BertModel.from_pretrained("google-bert/bert-base-uncased")

        num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
        print("We have added", num_added_toks, "tokens")
        # Note: resize_token_embeddings expects to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))
        ```"""
        added_tokens = 0
        if new_tokens is None:
            return added_tokens
        current_vocab = self.get_vocab().copy()
        new_idx = len(current_vocab)
        for token in new_tokens:
            if not isinstance(token, (str, AddedToken)):
                raise TypeError(f'Token {token} is not a string but a {type(token)}.')
            if str(token) == '':
                continue
            if isinstance(token, str):
                if token in self._added_tokens_encoder:
                    continue
                else:
                    is_special = token in self.all_special_tokens or special_tokens
                    token = AddedToken(token, rstrip=False, lstrip=False, normalized=not is_special, special=is_special)
            elif special_tokens:
                token.__setstate__({'special': True, 'normalized': token.normalized})
            if token in self._added_tokens_decoder:
                continue
            if not token.special and token.normalized and getattr(self, 'do_lower_case', False):
                token.content = token.content.lower()
            if token.content not in current_vocab:
                token_index = new_idx + added_tokens
                current_vocab[token.content] = token_index
                added_tokens += 1
            else:
                token_index = current_vocab[token.content]
            if token.special and str(token) not in self.all_special_tokens:
                self._additional_special_tokens.append(token)
            self._added_tokens_decoder[token_index] = token
            self._added_tokens_encoder[token.content] = token_index
            if self.verbose:
                logger.info(f'Adding {token} to the vocabulary')
        self._update_trie()
        return added_tokens

    def _update_trie(self, unique_no_split_tokens: Optional[str]=[]):
        for token in self._added_tokens_decoder.values():
            if token not in self.tokens_trie._tokens:
                self.tokens_trie.add(token.content)
        for token in unique_no_split_tokens:
            if token not in self.tokens_trie._tokens:
                self.tokens_trie.add(token)

    def num_special_tokens_to_add(self, pair: bool=False) -> int:
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        <Tip>

        This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
        this inside your training loop.

        </Tip>

        Args:
            pair (`bool`, *optional*, defaults to `False`):
                Whether the number of added tokens should be computed in the case of a sequence pair or a single
                sequence.

        Returns:
            `int`: Number of special tokens added to sequences.
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

    def tokenize(self, text: TextInput, **kwargs) -> List[str]:
        """
        Converts a string into a sequence of tokens, using the tokenizer.

        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
        (BPE/SentencePieces/WordPieces). Takes care of added tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            **kwargs (additional keyword arguments):
                Passed along to the model-specific `prepare_for_tokenization` preprocessing method.

        Returns:
            `List[str]`: The list of tokens.
        """
        split_special_tokens = kwargs.pop('split_special_tokens', self.split_special_tokens)
        text, kwargs = self.prepare_for_tokenization(text, **kwargs)
        if kwargs:
            logger.warning(f'Keyword arguments {kwargs} not recognized.')
        if hasattr(self, 'do_lower_case') and self.do_lower_case:
            escaped_special_toks = [re.escape(s_tok) for s_tok in self.all_special_tokens]
            escaped_special_toks += [re.escape(s_tok.content) for s_tok in self._added_tokens_decoder.values() if not s_tok.special and s_tok.normalized]
            pattern = '(' + '|'.join(escaped_special_toks) + ')|' + '(.+?)'
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)
        if split_special_tokens:
            no_split_token = []
            tokens = [text]
        else:
            no_split_token = self._added_tokens_encoder.keys()
            tokens = self.tokens_trie.split(text)
        for i, token in enumerate(tokens):
            if token in no_split_token:
                tok_extended = self._added_tokens_decoder.get(self._added_tokens_encoder[token], None)
                left = tokens[i - 1] if i > 0 else None
                right = tokens[i + 1] if i < len(tokens) - 1 else None
                if isinstance(tok_extended, AddedToken):
                    if tok_extended.rstrip and right:
                        tokens[i + 1] = right.lstrip()
                    if tok_extended.lstrip and left:
                        tokens[i - 1] = left.rstrip()
                    if tok_extended.single_word and left and (left[-1] != ' '):
                        tokens[i - 1] += token
                        tokens[i] = ''
                    elif tok_extended.single_word and right and (right[0] != ' '):
                        tokens[i + 1] = token + tokens[i + 1]
                        tokens[i] = ''
                else:
                    raise ValueError(f'{tok_extended} cannot be tokenized because it was not properly added to the tokenizer. This means that it is not an `AddedToken` but a {type(tok_extended)}')
        tokenized_text = []
        for token in tokens:
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize(token))
        return tokenized_text

    def _tokenize(self, text, **kwargs):
        """
        Converts a string into a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
        if tokens is None:
            return None
        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)
        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None
        if token in self._added_tokens_encoder:
            return self._added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):
        raise NotImplementedError

    def _encode_plus(self, text: Union[TextInput, PreTokenizedInput, EncodedInput], text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]]=None, add_special_tokens: bool=True, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, truncation_strategy: TruncationStrategy=TruncationStrategy.DO_NOT_TRUNCATE, max_length: Optional[int]=None, stride: int=0, is_split_into_words: bool=False, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = list(itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text)))
                    return self.convert_tokens_to_ids(tokens)
                else:
                    return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            elif is_split_into_words:
                raise ValueError(f'Input {text} is not valid. Should be a string or a list/tuple of strings when `is_split_into_words=True`.')
            else:
                raise ValueError(f'Input {text} is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers.')
        if return_offsets_mapping:
            raise NotImplementedError('return_offset_mapping is not available when using Python tokenizers. To use this feature, change your tokenizer to one deriving from transformers.PreTrainedTokenizerFast. More information on available tokenizers at https://github.com/huggingface/transformers/pull/2674')
        first_ids = get_input_ids(text)
        second_ids = get_input_ids(text_pair) if text_pair is not None else None
        return self.prepare_for_model(first_ids, pair_ids=second_ids, add_special_tokens=add_special_tokens, padding=padding_strategy.value, truncation=truncation_strategy.value, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, prepend_batch_axis=True, return_attention_mask=return_attention_mask, return_token_type_ids=return_token_type_ids, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_length=return_length, verbose=verbose)

    def _batch_encode_plus(self, batch_text_or_text_pairs: Union[List[TextInput], List[TextInputPair], List[PreTokenizedInput], List[PreTokenizedInputPair], List[EncodedInput], List[EncodedInputPair]], add_special_tokens: bool=True, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, truncation_strategy: TruncationStrategy=TruncationStrategy.DO_NOT_TRUNCATE, max_length: Optional[int]=None, stride: int=0, is_split_into_words: bool=False, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = list(itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text)))
                    return self.convert_tokens_to_ids(tokens)
                else:
                    return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError('Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers.')
        if return_offsets_mapping:
            raise NotImplementedError('return_offset_mapping is not available when using Python tokenizers. To use this feature, change your tokenizer to one deriving from transformers.PreTrainedTokenizerFast.')
        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = (ids_or_pair_ids, None)
            elif is_split_into_words and (not isinstance(ids_or_pair_ids[0], (list, tuple))):
                ids, pair_ids = (ids_or_pair_ids, None)
            else:
                ids, pair_ids = ids_or_pair_ids
            first_ids = get_input_ids(ids)
            second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids))
        batch_outputs = self._batch_prepare_for_model(input_ids, add_special_tokens=add_special_tokens, padding_strategy=padding_strategy, truncation_strategy=truncation_strategy, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask, return_token_type_ids=return_token_type_ids, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_length=return_length, return_tensors=return_tensors, verbose=verbose)
        return BatchEncoding(batch_outputs)

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def _batch_prepare_for_model(self, batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]], add_special_tokens: bool=True, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, truncation_strategy: TruncationStrategy=TruncationStrategy.DO_NOT_TRUNCATE, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[str]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_length: bool=False, verbose: bool=True) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
        """
        batch_outputs = {}
        for first_ids, second_ids in batch_ids_pairs:
            outputs = self.prepare_for_model(first_ids, second_ids, add_special_tokens=add_special_tokens, padding=PaddingStrategy.DO_NOT_PAD.value, truncation=truncation_strategy.value, max_length=max_length, stride=stride, pad_to_multiple_of=None, return_attention_mask=False, return_token_type_ids=return_token_type_ids, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_length=return_length, return_tensors=None, prepend_batch_axis=False, verbose=verbose)
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)
        batch_outputs = self.pad(batch_outputs, padding=padding_strategy.value, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask)
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)
        return batch_outputs

    def prepare_for_tokenization(self, text: str, is_split_into_words: bool=False, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Performs any necessary transformations before tokenization.

        This method should pop the arguments from kwargs and return the remaining `kwargs` as well. We test the
        `kwargs` at the end of the encoding process to be sure all the arguments have been used.

        Args:
            text (`str`):
                The text to prepare.
            is_split_into_words (`bool`, *optional*, defaults to `False`):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
            kwargs (`Dict[str, Any]`, *optional*):
                Keyword arguments to use for the tokenization.

        Returns:
            `Tuple[str, Dict[str, Any]]`: The prepared text and the unused kwargs.
        """
        return (text, kwargs)

    def get_special_tokens_mask(self, token_ids_0: List, token_ids_1: Optional[List]=None, already_has_special_tokens: bool=False) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError('You should not supply a second sequence if the provided sequence of ids is already formatted with special tokens for the model.')
            return super().get_special_tokens_mask(token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True)
        return [0] * ((len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))

    @overload
    def convert_ids_to_tokens(self, ids: int, skip_special_tokens: bool=False) -> str:
        ...

    @overload
    def convert_ids_to_tokens(self, ids: List[int], skip_special_tokens: bool=False) -> List[str]:
        ...

    def convert_ids_to_tokens(self, ids: Union[int, List[int]], skip_special_tokens: bool=False) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        if isinstance(ids, int):
            if ids in self._added_tokens_decoder:
                return self._added_tokens_decoder[ids].content
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in self._added_tokens_decoder:
                tokens.append(self._added_tokens_decoder[index].content)
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def _convert_id_to_token(self, index: int) -> str:
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return ' '.join(tokens)

    def _decode(self, token_ids: List[int], skip_special_tokens: bool=False, clean_up_tokenization_spaces: bool=None, spaces_between_special_tokens: bool=True, **kwargs) -> str:
        self._decode_use_source_tokenizer = kwargs.pop('use_source_tokenizer', False)
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        legacy_added_tokens = set(self._added_tokens_encoder.keys()) - set(self.all_special_tokens) | {token for token in self.additional_special_tokens if self.convert_tokens_to_ids(token) >= self.vocab_size}
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in legacy_added_tokens:
                if current_sub_text:
                    string = self.convert_tokens_to_string(current_sub_text)
                    if len(string) > 0:
                        sub_texts.append(string)
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))
        if spaces_between_special_tokens:
            text = ' '.join(sub_texts)
        else:
            text = ''.join(sub_texts)
        clean_up_tokenization_spaces = clean_up_tokenization_spaces if clean_up_tokenization_spaces is not None else self.clean_up_tokenization_spaces
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text