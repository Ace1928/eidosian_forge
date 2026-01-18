import contextlib
import io
import os
import sys
import shutil
import subprocess
import tempfile
from pyflakes.checker import PYPY
from pyflakes.messages import UnusedImport
from pyflakes.reporter import Reporter
from pyflakes.api import (
from pyflakes.test.harness import TestCase, skipIf
class SysStreamCapturing:
    """Context manager capturing sys.stdin, sys.stdout and sys.stderr.

    The file handles are replaced with a StringIO object.
    """

    def __init__(self, stdin):
        self._stdin = io.StringIO(stdin or '', newline=os.linesep)

    def __enter__(self):
        self._orig_stdin = sys.stdin
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdin = self._stdin
        sys.stdout = self._stdout_stringio = io.StringIO(newline=os.linesep)
        sys.stderr = self._stderr_stringio = io.StringIO(newline=os.linesep)
        return self

    def __exit__(self, *args):
        self.output = self._stdout_stringio.getvalue()
        self.error = self._stderr_stringio.getvalue()
        sys.stdin = self._orig_stdin
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr