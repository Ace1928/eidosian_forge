import os
import pytest
import sys
from tempfile import TemporaryFile
from numpy.distutils import exec_command
from numpy.distutils.exec_command import get_pythonexe
from numpy.testing import tempdir, assert_, assert_warns, IS_WASM
from io import StringIO
class redirect_stderr:
    """Context manager to redirect stderr for exec_command test."""

    def __init__(self, stderr=None):
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stderr = sys.stderr
        sys.stderr = self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stderr.flush()
        sys.stderr = self.old_stderr
        self._stderr.close()