from __future__ import annotations
import enum
import sys
import types
import typing
import warnings
def _check_byteslike(name: str, value: bytes) -> None:
    try:
        memoryview(value)
    except TypeError:
        raise TypeError(f'{name} must be bytes-like')