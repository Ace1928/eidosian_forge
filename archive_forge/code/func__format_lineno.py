import os
import sys
from io import open as io_open
import fnmatch
from IPython.core.error import StdinNotImplementedError
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core.magic_arguments import (argument, magic_arguments,
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils import io
def _format_lineno(session, line):
    """Helper function to format line numbers properly."""
    if session in (0, history_manager.session_number):
        return str(line)
    return '%s/%s' % (session, line)