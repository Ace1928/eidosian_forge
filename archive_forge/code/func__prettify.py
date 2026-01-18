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
def _prettify(self, message_dict):
    """Reorders items in a MessageDict such that it is more readable."""
    for key in self._prettify_order:
        if key not in message_dict:
            continue
        value = message_dict[key]
        del message_dict[key]
        message_dict[key] = value