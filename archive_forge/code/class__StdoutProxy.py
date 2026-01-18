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
class _StdoutProxy(object):
    """
    Proxy for stdout, as returned by
    :class:`CommandLineInterface.stdout_proxy`.
    """

    def __init__(self, cli, raw=False):
        assert isinstance(cli, CommandLineInterface)
        assert isinstance(raw, bool)
        self._lock = threading.RLock()
        self._cli = cli
        self._raw = raw
        self._buffer = []
        self.errors = sys.__stdout__.errors
        self.encoding = sys.__stdout__.encoding

    def _do(self, func):
        if self._cli._is_running:
            run_in_terminal = functools.partial(self._cli.run_in_terminal, func)
            self._cli.eventloop.call_from_executor(run_in_terminal)
        else:
            func()

    def _write(self, data):
        """
        Note: print()-statements cause to multiple write calls.
              (write('line') and write('
')). Of course we don't want to call
              `run_in_terminal` for every individual call, because that's too
              expensive, and as long as the newline hasn't been written, the
              text itself is again overwritter by the rendering of the input
              command line. Therefor, we have a little buffer which holds the
              text until a newline is written to stdout.
        """
        if '\n' in data:
            before, after = data.rsplit('\n', 1)
            to_write = self._buffer + [before, '\n']
            self._buffer = [after]

            def run():
                for s in to_write:
                    if self._raw:
                        self._cli.output.write_raw(s)
                    else:
                        self._cli.output.write(s)
            self._do(run)
        else:
            self._buffer.append(data)

    def write(self, data):
        with self._lock:
            self._write(data)

    def _flush(self):

        def run():
            for s in self._buffer:
                if self._raw:
                    self._cli.output.write_raw(s)
                else:
                    self._cli.output.write(s)
            self._buffer = []
            self._cli.output.flush()
        self._do(run)

    def flush(self):
        """
        Flush buffered output.
        """
        with self._lock:
            self._flush()