import argparse
import os
import sys
from collections.abc import Mapping
from typing import Callable, Dict, List, Optional, Sequence, Union
from . import io as _io
from .completers import ChoicesCompleter, FilesCompleter, SuppressCompleter
from .io import debug, mute_stderr
from .lexers import split_line
from .packages._argparse import IntrospectiveArgumentParser, action_is_greedy, action_is_open, action_is_satisfied
def _init_debug_stream(self):
    """Initialize debug output stream

        By default, writes to file descriptor 9, or stderr if that fails.
        This can be overridden by derived classes, for example to avoid
        clashes with file descriptors being used elsewhere (such as in pytest).
        """
    try:
        _io.debug_stream = os.fdopen(9, 'w')
    except Exception:
        _io.debug_stream = sys.stderr
    debug()