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
class DropUnderscoreHeaderReader(HeaderReader):
    """Custom HeaderReader to exclude any headers with underscores in them."""

    def _allow_header(self, key_name):
        orig = super(DropUnderscoreHeaderReader, self)._allow_header(key_name)
        return orig and '_' not in key_name