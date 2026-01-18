import sys
import os
import time
import locale
import signal
import urwid
from typing import Optional
from . import args as bpargs, repl, translations
from .formatter import theme_map
from .translations import _
from .keys import urwid_key_dispatch as key_dispatch
class BPythonListBox(urwid.ListBox):
    """Like `urwid.ListBox`, except that it does not eat up and
    down keys.
    """

    def keypress(self, size, key):
        if key not in ('up', 'down'):
            return urwid.ListBox.keypress(self, size, key)
        return key