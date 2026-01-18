import abc
import collections
import contextlib
import io
from io import UnsupportedOperation
import os
import sys
from tempfile import TemporaryFile
from types import TracebackType
from typing import Any
from typing import AnyStr
from typing import BinaryIO
from typing import Final
from typing import final
from typing import Generator
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Literal
from typing import NamedTuple
from typing import Optional
from typing import TextIO
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from _pytest.config import Config
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import SubRequest
from _pytest.nodes import Collector
from _pytest.nodes import File
from _pytest.nodes import Item
from _pytest.reports import CollectReport
def _windowsconsoleio_workaround(stream: TextIO) -> None:
    """Workaround for Windows Unicode console handling.

    Python 3.6 implemented Unicode console handling for Windows. This works
    by reading/writing to the raw console handle using
    ``{Read,Write}ConsoleW``.

    The problem is that we are going to ``dup2`` over the stdio file
    descriptors when doing ``FDCapture`` and this will ``CloseHandle`` the
    handles used by Python to write to the console. Though there is still some
    weirdness and the console handle seems to only be closed randomly and not
    on the first call to ``CloseHandle``, or maybe it gets reopened with the
    same handle value when we suspend capturing.

    The workaround in this case will reopen stdio with a different fd which
    also means a different handle by replicating the logic in
    "Py_lifecycle.c:initstdio/create_stdio".

    :param stream:
        In practice ``sys.stdout`` or ``sys.stderr``, but given
        here as parameter for unittesting purposes.

    See https://github.com/pytest-dev/py/issues/103.
    """
    if not sys.platform.startswith('win32') or hasattr(sys, 'pypy_version_info'):
        return
    if not hasattr(stream, 'buffer'):
        return
    buffered = hasattr(stream.buffer, 'raw')
    raw_stdout = stream.buffer.raw if buffered else stream.buffer
    if not isinstance(raw_stdout, io._WindowsConsoleIO):
        return

    def _reopen_stdio(f, mode):
        if not buffered and mode[0] == 'w':
            buffering = 0
        else:
            buffering = -1
        return io.TextIOWrapper(open(os.dup(f.fileno()), mode, buffering), f.encoding, f.errors, f.newlines, f.line_buffering)
    sys.stdin = _reopen_stdio(sys.stdin, 'rb')
    sys.stdout = _reopen_stdio(sys.stdout, 'wb')
    sys.stderr = _reopen_stdio(sys.stderr, 'wb')