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
class _PatchStdoutContext(object):

    def __init__(self, new_stdout, patch_stdout=True, patch_stderr=True):
        self.new_stdout = new_stdout
        self.patch_stdout = patch_stdout
        self.patch_stderr = patch_stderr

    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        if self.patch_stdout:
            sys.stdout = self.new_stdout
        if self.patch_stderr:
            sys.stderr = self.new_stdout

    def __exit__(self, *a, **kw):
        if self.patch_stdout:
            sys.stdout = self.original_stdout
        if self.patch_stderr:
            sys.stderr = self.original_stderr