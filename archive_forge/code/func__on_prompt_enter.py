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
def _on_prompt_enter(self, edit, new_text):
    """Reset the statusbar and pass the input from the prompt to the caller
        via 'prompt_result'."""
    self.settext(self.s)
    urwid.emit_signal(self, 'prompt_result', new_text)