from __future__ import annotations
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import AbstractSet, Collection, Literal, NoReturn, Optional, Union
import regex
from tiktoken import _tiktoken
def _encode_only_native_bpe(self, text: str) -> list[int]:
    """Encodes a string into tokens, but do regex splitting in Python."""
    _unused_pat = regex.compile(self._pat_str)
    ret = []
    for piece in regex.findall(_unused_pat, text):
        ret.extend(self._core_bpe.encode_single_piece(piece))
    return ret