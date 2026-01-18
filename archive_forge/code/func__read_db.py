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
def _read_db(self) -> None:
    """Read the metadata from a database so that we are ready to use it."""
    with self._dbs[threading.get_ident()] as db:
        try:
            row = db.execute_one('select version from coverage_schema')
            assert row is not None
        except Exception as exc:
            if 'no such table: coverage_schema' in str(exc):
                self._init_db(db)
            else:
                raise DataError("Data file {!r} doesn't seem to be a coverage data file: {}".format(self._filename, exc)) from exc
        else:
            schema_version = row[0]
            if schema_version != SCHEMA_VERSION:
                raise DataError("Couldn't use data file {!r}: wrong schema: {} instead of {}".format(self._filename, schema_version, SCHEMA_VERSION))
        row = db.execute_one("select value from meta where key = 'has_arcs'")
        if row is not None:
            self._has_arcs = bool(int(row[0]))
            self._has_lines = not self._has_arcs
        with db.execute('select id, path from file') as cur:
            for file_id, path in cur:
                self._file_map[path] = file_id