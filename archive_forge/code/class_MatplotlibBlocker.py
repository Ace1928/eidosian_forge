import sys
import unittest
from numba.tests.support import captured_stdout
from numba.core.config import IS_WIN32
class MatplotlibBlocker:
    """Blocks the import of matplotlib, so that doc examples that attempt to
    plot the output don't result in plots popping up and blocking testing."""

    def find_spec(self, fullname, path, target=None):
        if fullname == 'matplotlib':
            msg = 'Blocked import of matplotlib for test suite run'
            raise ImportError(msg)