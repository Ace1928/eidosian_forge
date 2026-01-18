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
def _invalid_if_no_key(func):

    def wrap(self, key, *args, **kwargs):
        try:
            return func(self, key, *args, **kwargs)
        except KeyError:
            message = Message if self.message is None else self.message
            raise message.isnt_valid('missing property {0!r}', key)
    return wrap