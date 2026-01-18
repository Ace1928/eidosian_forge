from __future__ import annotations
import collections
import datetime
import functools
import glob
import itertools
import os
import random
import socket
import sqlite3
import string
import sys
import textwrap
import threading
import zlib
from typing import (
from coverage.debug import NoDebugging, auto_repr
from coverage.exceptions import CoverageException, DataError
from coverage.files import PathAliases
from coverage.misc import file_be_gone, isolate_module
from coverage.numbits import numbits_to_nums, numbits_union, nums_to_numbits
from coverage.sqlitedb import SqliteDb
from coverage.types import AnyCallable, FilePath, TArc, TDebugCtl, TLineNo, TWarnFn
from coverage.version import __version__
def _start_using(self) -> None:
    """Call this before using the database at all."""
    if self._pid != os.getpid():
        self._reset()
        self._choose_filename()
        self._pid = os.getpid()
    if not self._have_used:
        self.erase()
    self._have_used = True