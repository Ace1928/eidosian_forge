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
class MessageHandlers(object):
    """A simple delegating message handlers object for use with JsonMessageChannel.
    For every argument provided, the object gets an attribute with the corresponding
    name and value.
    """

    def __init__(self, **kwargs):
        for name, func in kwargs.items():
            setattr(self, name, func)