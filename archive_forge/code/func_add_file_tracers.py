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
@_locked
def add_file_tracers(self, file_tracers: Mapping[str, str]) -> None:
    """Add per-file plugin information.

        `file_tracers` is { filename: plugin_name, ... }

        """
    if self._debug.should('dataop'):
        self._debug.write('Adding file tracers: %d files' % (len(file_tracers),))
    if not file_tracers:
        return
    self._start_using()
    with self._connect() as con:
        for filename, plugin_name in file_tracers.items():
            file_id = self._file_id(filename, add=True)
            existing_plugin = self.file_tracer(filename)
            if existing_plugin:
                if existing_plugin != plugin_name:
                    raise DataError("Conflicting file tracer name for '{}': {!r} vs {!r}".format(filename, existing_plugin, plugin_name))
            elif plugin_name:
                con.execute_void('insert into tracer (file_id, tracer) values (?, ?)', (file_id, plugin_name))