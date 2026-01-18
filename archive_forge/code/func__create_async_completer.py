from __future__ import unicode_literals
import functools
import os
import signal
import six
import sys
import textwrap
import threading
import time
import types
import weakref
from subprocess import Popen
from .application import Application, AbortAction
from .buffer import Buffer
from .buffer_mapping import BufferMapping
from .completion import CompleteEvent, get_common_complete_suffix
from .enums import SEARCH_BUFFER
from .eventloop.base import EventLoop
from .eventloop.callbacks import EventLoopCallbacks
from .filters import Condition
from .input import StdinInput, Input
from .key_binding.input_processor import InputProcessor
from .key_binding.input_processor import KeyPress
from .key_binding.registry import Registry
from .key_binding.vi_state import ViState
from .keys import Keys
from .output import Output
from .renderer import Renderer, print_tokens
from .search_state import SearchState
from .utils import Event
from .buffer import AcceptAction
def _create_async_completer(self, buffer):
    """
        Create function for asynchronous autocompletion.
        (Autocomplete in other thread.)
        """
    complete_thread_running = [False]

    def completion_does_nothing(document, completion):
        """
            Return `True` if applying this completion doesn't have any effect.
            (When it doesn't insert any new text.
            """
        text_before_cursor = document.text_before_cursor
        replaced_text = text_before_cursor[len(text_before_cursor) + completion.start_position:]
        return replaced_text == completion.text

    def async_completer(select_first=False, select_last=False, insert_common_part=False, complete_event=None):
        document = buffer.document
        complete_event = complete_event or CompleteEvent(text_inserted=True)
        if complete_thread_running[0]:
            return
        if buffer.complete_state or not buffer.completer:
            return
        complete_thread_running[0] = True

        def run():
            completions = list(buffer.completer.get_completions(document, complete_event))

            def callback():
                """
                    Set the new complete_state in a safe way. Don't replace an
                    existing complete_state if we had one. (The user could have
                    pressed 'Tab' in the meantime. Also don't set it if the text
                    was changed in the meantime.
                    """
                complete_thread_running[0] = False
                if len(completions) == 1 and completion_does_nothing(document, completions[0]):
                    del completions[:]
                if buffer.text == document.text and buffer.cursor_position == document.cursor_position and (not buffer.complete_state):
                    set_completions = True
                    select_first_anyway = False
                    if insert_common_part:
                        common_part = get_common_complete_suffix(document, completions)
                        if common_part:
                            buffer.insert_text(common_part)
                            if len(completions) > 1:
                                completions[:] = [c.new_completion_from_position(len(common_part)) for c in completions]
                            else:
                                set_completions = False
                        elif len(completions) == 1:
                            select_first_anyway = True
                    if set_completions:
                        buffer.set_completions(completions=completions, go_to_first=select_first or select_first_anyway, go_to_last=select_last)
                    self.invalidate()
                elif not buffer.complete_state:
                    async_completer()
            if self.eventloop:
                self.eventloop.call_from_executor(callback)
        self.eventloop.run_in_executor(run)
    return async_completer