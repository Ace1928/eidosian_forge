from __future__ import annotations
import collections
import contextlib
import functools
import itertools
import os
import socket
import sys
import threading
from debugpy.common import json, log, util
from debugpy.common.util import hide_thread_from_debugger
class NoMoreMessages(JsonIOError, EOFError):
    """Indicates that there are no more messages that can be read from or written
    to a stream.
    """

    def __init__(self, *args, **kwargs):
        args = args if len(args) else ['No more messages']
        super().__init__(*args, **kwargs)