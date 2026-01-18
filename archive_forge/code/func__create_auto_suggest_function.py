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
def _create_auto_suggest_function(self, buffer):
    """
        Create function for asynchronous auto suggestion.
        (AutoSuggest in other thread.)
        """
    suggest_thread_running = [False]

    def async_suggestor():
        document = buffer.document
        if suggest_thread_running[0]:
            return
        if buffer.suggestion or not buffer.auto_suggest:
            return
        suggest_thread_running[0] = True

        def run():
            suggestion = buffer.auto_suggest.get_suggestion(self, buffer, document)

            def callback():
                suggest_thread_running[0] = False
                if buffer.text == document.text and buffer.cursor_position == document.cursor_position:
                    buffer.suggestion = suggestion
                    self.invalidate()
                else:
                    async_suggestor()
            if self.eventloop:
                self.eventloop.call_from_executor(callback)
        self.eventloop.run_in_executor(run)
    return async_suggestor