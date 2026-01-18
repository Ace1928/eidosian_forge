import contextlib
import errno
import itertools
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import unicodedata
from enum import Enum
from types import FrameType, TracebackType
from typing import (
from .._typing_compat import Literal
import greenlet
from curtsies import (
from curtsies.configfile_keynames import keymap as key_dispatch
from curtsies.input import is_main_thread
from curtsies.window import CursorAwareWindow
from cwcwidth import wcswidth
from pygments import format as pygformat
from pygments.formatters import TerminalFormatter
from pygments.lexers import Python3Lexer
from . import events as bpythonevents, sitefix, replpainter as paint
from ..config import Config
from .coderunner import (
from .filewatch import ModuleChangedEventHandler
from .interaction import StatusBar
from .interpreter import (
from .manual_readline import (
from .parse import parse as bpythonparse, func_for_letter, color_for_letter
from .preprocess import preprocess
from .. import __version__
from ..config import getpreferredencoding
from ..formatter import BPythonFormatter
from ..pager import get_pager_command
from ..repl import (
from ..translations import _
from ..line import CHARACTER_PAIR_MAP
class ReevaluateFakeStdin:
    """Stdin mock used during reevaluation (undo) so raw_inputs don't have to
    be reentered"""

    def __init__(self, fakestdin: FakeStdin, repl: 'BaseRepl'):
        self.fakestdin = fakestdin
        self.repl = repl
        self.readline_results = fakestdin.readline_results[:]

    def readline(self):
        if self.readline_results:
            value = self.readline_results.pop(0)
        else:
            value = 'no saved input available'
        self.repl.send_to_stdouterr(value)
        return value