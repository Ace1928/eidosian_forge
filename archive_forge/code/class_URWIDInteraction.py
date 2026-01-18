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
class URWIDInteraction(repl.Interaction):

    def __init__(self, config, statusbar, frame):
        super().__init__(config)
        self.statusbar = statusbar
        self.frame = frame
        urwid.connect_signal(statusbar, 'prompt_result', self._prompt_result)
        self.callback = None

    def confirm(self, q, callback):
        """Ask for yes or no and call callback to return the result"""

        def callback_wrapper(result):
            callback(result.lower() in (_('y'), _('yes')))
        self.prompt(q, callback_wrapper, single=True)

    def notify(self, s, n=10, wait_for_keypress=False):
        return self.statusbar.message(s, n)

    def prompt(self, s, callback=None, single=False):
        """Prompt the user for input. The result will be returned via calling
        callback. Note that there can only be one prompt active. But the
        callback can already start a new prompt."""
        if self.callback is not None:
            raise Exception('Prompt already in progress')
        self.callback = callback
        self.statusbar.prompt(s, single=single)
        self.frame.set_focus('footer')

    def _prompt_result(self, text):
        self.frame.set_focus('body')
        if self.callback is not None:
            callback = self.callback
            self.callback = None
            callback(text)

    def file_prompt(self, s: str) -> Optional[str]:
        raise NotImplementedError