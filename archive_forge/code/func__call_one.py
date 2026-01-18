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
def _call_one(self, text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]], text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy]=None, max_length: Optional[int]=None, stride: int=0, is_split_into_words: bool=False, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:

    def _is_valid_text_input(t):
        if isinstance(t, str):
            return True
        elif isinstance(t, (list, tuple)):
            if len(t) == 0:
                return True
            elif isinstance(t[0], str):
                return True
            elif isinstance(t[0], (list, tuple)):
                return len(t[0]) == 0 or isinstance(t[0][0], str)
            else:
                return False
        else:
            return False
    if not _is_valid_text_input(text):
        raise ValueError('text input must be of type `str` (single example), `List[str]` (batch or single pretokenized example) or `List[List[str]]` (batch of pretokenized examples).')
    if text_pair is not None and (not _is_valid_text_input(text_pair)):
        raise ValueError('text input must be of type `str` (single example), `List[str]` (batch or single pretokenized example) or `List[List[str]]` (batch of pretokenized examples).')
    if is_split_into_words:
        is_batched = isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple))
    else:
        is_batched = isinstance(text, (list, tuple))
    if is_batched:
        if isinstance(text_pair, str):
            raise TypeError('when tokenizing batches of text, `text_pair` must be a list or tuple with the same length as `text`.')
        if text_pair is not None and len(text) != len(text_pair):
            raise ValueError(f'batch length of `text`: {len(text)} does not match batch length of `text_pair`: {len(text_pair)}.')
        batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
        return self.batch_encode_plus(batch_text_or_text_pairs=batch_text_or_text_pairs, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, stride=stride, is_split_into_words=is_split_into_words, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, **kwargs)
    else:
        return self.encode_plus(text=text, text_pair=text_pair, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, stride=stride, is_split_into_words=is_split_into_words, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, **kwargs)