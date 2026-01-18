import builtins
import codecs
import enum
import io
import json
import os
import types
import typing
from typing import (
import attr
@attr.s(auto_attribs=True, repr=False)
class ReaderWriterBase:
    """
    Base class with shared behaviour for both the reader and writer.
    """
    _fp: Union[typing.IO[str], typing.IO[bytes], None] = attr.ib(default=None, init=False)
    _closed: bool = attr.ib(default=False, init=False)
    _should_close_fp: bool = attr.ib(default=False, init=False)

    def close(self) -> None:
        """
        Close this reader/writer.

        This closes the underlying file if that file has been opened by
        this reader/writer. When an already opened file-like object was
        provided, the caller is responsible for closing it.
        """
        if self._closed:
            return
        self._closed = True
        if self._fp is not None and self._should_close_fp:
            self._fp.close()

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        wrapped = self._repr_for_wrapped()
        return f'<jsonlines.{cls_name} at 0x{id(self):x} wrapping {wrapped}>'

    def _repr_for_wrapped(self) -> str:
        raise NotImplementedError

    def __enter__(self: TRW) -> TRW:
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[types.TracebackType]) -> None:
        self.close()