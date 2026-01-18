import codecs
import io
import os
import sys
import warnings
from ..lazy_import import lazy_import
import time
from breezy import (
from .. import config, osutils, trace
from . import NullProgressView, UIFactory
def _setup_mode(self):
    """Setup input mode (line-based, char-based) and echo-back.

        Line-based input is used if the BRZ_TEXTUI_INPUT environment
        variable is set to 'line-based', or if there is no controlling
        terminal.
        """
    is_tty = self.ui.raw_stdin.isatty()
    if os.environ.get('BRZ_TEXTUI_INPUT') != 'line-based' and self.ui.raw_stdin == _unwrap_stream(sys.stdin) and is_tty:
        self.line_based = False
        self.echo_back = True
    else:
        self.line_based = True
        self.echo_back = not is_tty