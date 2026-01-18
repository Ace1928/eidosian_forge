import asyncio
import builtins
import gc
import getpass
import os
import signal
import sys
import threading
import typing as t
from contextlib import contextmanager
from functools import partial
import comm
from IPython.core import release
from IPython.utils.tokenutil import line_at_cursor, token_at_cursor
from jupyter_client.session import extract_header
from traitlets import Any, Bool, HasTraits, Instance, List, Type, observe, observe_compat
from zmq.eventloop.zmqstream import ZMQStream
from .comm.comm import BaseComm
from .comm.manager import CommManager
from .compiler import XCachingCompiler
from .eventloops import _use_appnope
from .iostream import OutStream
from .kernelbase import Kernel as KernelBase
from .kernelbase import _accepts_parameters
from .zmqshell import ZMQInteractiveShell
def _experimental_do_complete(self, code, cursor_pos):
    """
        Experimental completions from IPython, using Jedi.
        """
    if cursor_pos is None:
        cursor_pos = len(code)
    with _provisionalcompleter():
        assert self.shell is not None
        raw_completions = self.shell.Completer.completions(code, cursor_pos)
        completions = list(_rectify_completions(code, raw_completions))
        comps = []
        for comp in completions:
            comps.append(dict(start=comp.start, end=comp.end, text=comp.text, type=comp.type, signature=comp.signature))
    if completions:
        s = completions[0].start
        e = completions[0].end
        matches = [c.text for c in completions]
    else:
        s = cursor_pos
        e = cursor_pos
        matches = []
    return {'matches': matches, 'cursor_end': e, 'cursor_start': s, 'metadata': {_EXPERIMENTAL_KEY_NAME: comps}, 'status': 'ok'}