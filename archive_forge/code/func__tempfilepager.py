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
def _tempfilepager(generator: t.Iterable[str], cmd: str, color: t.Optional[bool]) -> None:
    """Page through text by invoking a program on a temporary file."""
    import tempfile
    fd, filename = tempfile.mkstemp()
    text = ''.join(generator)
    if not color:
        text = strip_ansi(text)
    encoding = get_best_encoding(sys.stdout)
    with open_stream(filename, 'wb')[0] as f:
        f.write(text.encode(encoding))
    try:
        os.system(f'{cmd} "{filename}"')
    finally:
        os.close(fd)
        os.unlink(filename)