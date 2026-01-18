from __future__ import annotations
import contextlib
import typing
class RequestNotRead(StreamError):
    """
    Attempted to access streaming request content, without having called `read()`.
    """

    def __init__(self) -> None:
        message = 'Attempted to access streaming request content, without having called `read()`.'
        super().__init__(message)