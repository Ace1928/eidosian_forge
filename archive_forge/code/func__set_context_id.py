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
def _set_context_id(self) -> None:
    """Use the _current_context to set _current_context_id."""
    context = self._current_context or ''
    context_id = self._context_id(context)
    if context_id is not None:
        self._current_context_id = context_id
    else:
        with self._connect() as con:
            self._current_context_id = con.execute_for_rowid('insert into context (context) values (?)', (context,))