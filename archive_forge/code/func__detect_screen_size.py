import os
import io
import re
import sys
import tempfile
import subprocess
from io import UnsupportedOperation
from pathlib import Path
from IPython import get_ipython
from IPython.display import display
from IPython.core.error import TryNext
from IPython.utils.data import chop
from IPython.utils.process import system
from IPython.utils.terminal import get_terminal_size
from IPython.utils import py3compat
def _detect_screen_size(screen_lines_def):
    """Attempt to work out the number of lines on the screen.

    This is called by page(). It can raise an error (e.g. when run in the
    test suite), so it's separated out so it can easily be called in a try block.
    """
    TERM = os.environ.get('TERM', None)
    if not ((TERM == 'xterm' or TERM == 'xterm-color') and sys.platform != 'sunos5'):
        return screen_lines_def
    try:
        import termios
        import curses
    except ImportError:
        return screen_lines_def
    try:
        term_flags = termios.tcgetattr(sys.stdout)
    except termios.error as err:
        raise TypeError('termios error: {0}'.format(err)) from err
    try:
        scr = curses.initscr()
    except AttributeError:
        return screen_lines_def
    screen_lines_real, screen_cols = scr.getmaxyx()
    curses.endwin()
    termios.tcsetattr(sys.stdout, termios.TCSANOW, term_flags)
    return screen_lines_real