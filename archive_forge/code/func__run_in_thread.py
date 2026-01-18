import os
import io
import re
import email.utils
import socket
import sys
import time
import traceback as traceback_
import logging
import platform
import queue
import contextlib
import threading
import urllib.parse
from functools import lru_cache
from . import connections, errors, __version__
from ._compat import bton
from ._compat import IS_PPC
from .workers import threadpool
from .makefile import MakeFile, StreamWriter
@contextlib.contextmanager
def _run_in_thread(self):
    """Context manager for running this server in a thread."""
    self.prepare()
    thread = threading.Thread(target=self.serve)
    thread.daemon = True
    thread.start()
    try:
        yield thread
    finally:
        self.stop()