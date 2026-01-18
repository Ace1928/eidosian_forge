import builtins
import os
import sys
import platform
from tempfile import NamedTemporaryFile
from textwrap import dedent
from unittest.mock import patch
from IPython.core import debugger
from IPython.testing import IPYTHON_TESTING_TIMEOUT_SCALE
from IPython.testing.decorators import skip_win32
import pytest
class _FakeInput(object):
    """
    A fake input stream for pdb's interactive debugger.  Whenever a
    line is read, print it (to simulate the user typing it), and then
    return it.  The set of lines to return is specified in the
    constructor; they should not have trailing newlines.
    """

    def __init__(self, lines):
        self.lines = iter(lines)

    def readline(self):
        line = next(self.lines)
        print(line)
        return line + '\n'