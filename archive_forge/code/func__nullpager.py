import time of Click down, some infrequently used functionality is
import contextlib
import math
import os
import sys
import time
import typing as t
from gettext import gettext as _
from io import StringIO
from types import TracebackType
from ._compat import _default_text_stdout
from ._compat import CYGWIN
from ._compat import get_best_encoding
from ._compat import isatty
from ._compat import open_stream
from ._compat import strip_ansi
from ._compat import term_len
from ._compat import WIN
from .exceptions import ClickException
from .utils import echo
def _nullpager(stream: t.TextIO, generator: t.Iterable[str], color: t.Optional[bool]) -> None:
    """Simply print unformatted text.  This is the ultimate fallback."""
    for text in generator:
        if not color:
            text = strip_ansi(text)
        stream.write(text)