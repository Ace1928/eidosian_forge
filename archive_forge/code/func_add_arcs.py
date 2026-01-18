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
def add_arcs(self, arc_data: Mapping[str, Collection[TArc]]) -> None:
    """Add measured arc data.

        `arc_data` is a dictionary mapping file names to iterables of pairs of
        ints::

            { filename: { (l1,l2), (l1,l2), ... }, ...}

        """
    if self._debug.should('dataop'):
        self._debug.write('Adding arcs: %d files, %d arcs total' % (len(arc_data), sum((len(arcs) for arcs in arc_data.values()))))
        if self._debug.should('dataop2'):
            for filename, arcs in sorted(arc_data.items()):
                self._debug.write(f'  {filename}: {arcs}')
    self._start_using()
    self._choose_lines_or_arcs(arcs=True)
    if not arc_data:
        return
    with self._connect() as con:
        self._set_context_id()
        for filename, arcs in arc_data.items():
            if not arcs:
                continue
            file_id = self._file_id(filename, add=True)
            data = [(file_id, self._current_context_id, fromno, tono) for fromno, tono in arcs]
            con.executemany_void('insert or ignore into arc ' + '(file_id, context_id, fromno, tono) values (?, ?, ?, ?)', data)