import os
import re
import sys
import typing as t
from functools import update_wrapper
from types import ModuleType
from types import TracebackType
from ._compat import _default_text_stderr
from ._compat import _default_text_stdout
from ._compat import _find_binary_writer
from ._compat import auto_wrap_for_ansi
from ._compat import binary_streams
from ._compat import open_stream
from ._compat import should_strip_ansi
from ._compat import strip_ansi
from ._compat import text_streams
from ._compat import WIN
from .globals import resolve_color_default
class KeepOpenFile:

    def __init__(self, file: t.IO[t.Any]) -> None:
        self._file: t.IO[t.Any] = file

    def __getattr__(self, name: str) -> t.Any:
        return getattr(self._file, name)

    def __enter__(self) -> 'KeepOpenFile':
        return self

    def __exit__(self, exc_type: t.Optional[t.Type[BaseException]], exc_value: t.Optional[BaseException], tb: t.Optional[TracebackType]) -> None:
        pass

    def __repr__(self) -> str:
        return repr(self._file)

    def __iter__(self) -> t.Iterator[t.AnyStr]:
        return iter(self._file)