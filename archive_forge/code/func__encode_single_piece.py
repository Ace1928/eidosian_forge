from __future__ import annotations
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import AbstractSet, Collection, Literal, NoReturn, Optional, Union
import regex
from tiktoken import _tiktoken
def _encode_single_piece(self, text_or_bytes: Union[str, bytes]) -> list[int]:
    """Encodes text corresponding to bytes without a regex split.

        NOTE: this will not encode any special tokens.

        ```
        >>> enc.encode_single_piece("helloqqqq")
        [31373, 38227, 38227]
        ```
        """
    if isinstance(text_or_bytes, str):
        text_or_bytes = text_or_bytes.encode('utf-8')
    return self._core_bpe.encode_single_piece(text_or_bytes)