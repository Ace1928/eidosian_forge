from __future__ import unicode_literals
from ctypes import windll, pointer
from ctypes.wintypes import DWORD
from six.moves import range
from prompt_toolkit.key_binding.input_processor import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.win32_types import EventTypes, KEY_EVENT_RECORD, MOUSE_EVENT_RECORD, INPUT_RECORD, STD_INPUT_HANDLE
import msvcrt
import os
import sys
import six
@staticmethod
def _is_paste(keys):
    """
        Return `True` when we should consider this list of keys as a paste
        event. Pasted text on windows will be turned into a
        `Keys.BracketedPaste` event. (It's not 100% correct, but it is probably
        the best possible way to detect pasting of text and handle that
        correctly.)
        """
    text_count = 0
    newline_count = 0
    for k in keys:
        if isinstance(k.key, six.text_type):
            text_count += 1
        if k.key == Keys.ControlJ:
            newline_count += 1
    return newline_count >= 1 and text_count > 1