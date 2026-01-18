import asyncio
import binascii
from collections import defaultdict
import contextlib
import errno
import functools
import importlib
import inspect
import json
import logging
import multiprocessing
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlencode, unquote, urlparse, parse_qsl, urlunparse
import warnings
from inspect import signature
from pathlib import Path
from subprocess import list2cmdline
from typing import (
import psutil
from google.protobuf import json_format
import ray
import ray._private.ray_constants as ray_constants
from ray.core.generated.runtime_env_common_pb2 import (
class DeferSigint(contextlib.AbstractContextManager):
    """Context manager that defers SIGINT signals until the the context is left."""

    def __init__(self):
        self.task_cancelled = False
        self.orig_sigint_handler = None
        self.orig_signal = None

    @classmethod
    def create_if_main_thread(cls) -> contextlib.AbstractContextManager:
        """Creates a DeferSigint context manager if running on the main thread,
        returns a no-op context manager otherwise.
        """
        if threading.current_thread() == threading.main_thread():
            return cls()
        else:
            return contextlib.nullcontext()

    def _set_task_cancelled(self, signum, frame):
        """SIGINT handler that defers the signal."""
        self.task_cancelled = True

    def _signal_monkey_patch(self, signum, handler):
        """Monkey patch for signal.signal that raises an error if a SIGINT handler is
        registered within the DeferSigint context.
        """
        if threading.current_thread() == threading.main_thread() and signum == signal.SIGINT:
            raise ValueError("Can't set signal handler for SIGINT while SIGINT is being deferred within a DeferSigint context.")
        return self.orig_signal(signum, handler)

    def __enter__(self):
        self.orig_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._set_task_cancelled)
        self.orig_signal = signal.signal
        signal.signal = self._signal_monkey_patch
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        assert self.orig_sigint_handler is not None
        assert self.orig_signal is not None
        signal.signal = self.orig_signal
        signal.signal(signal.SIGINT, self.orig_sigint_handler)
        if exc_type is None and self.task_cancelled:
            raise KeyboardInterrupt
        else:
            return False