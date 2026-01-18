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
class StatusbarEdit(urwid.Edit):
    """Wrapper around urwid.Edit used for the prompt in Statusbar.

    This class only adds a single signal that is emitted if the user presses
    Enter."""
    signals = urwid.Edit.signals + ['prompt_enter']

    def __init__(self, *args, **kwargs):
        self.single = False
        super().__init__(*args, **kwargs)

    def keypress(self, size, key):
        if self.single:
            urwid.emit_signal(self, 'prompt_enter', self, key)
        elif key == 'enter':
            urwid.emit_signal(self, 'prompt_enter', self, self.get_edit_text())
        else:
            return super().keypress(size, key)