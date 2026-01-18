from bisect import bisect_left
from bisect import bisect_right
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from inspect import isclass
import calendar
import collections
import datetime
import decimal
import hashlib
import itertools
import logging
import operator
import re
import socket
import struct
import sys
import threading
import time
import uuid
import warnings
class _WriteQuery(Query):

    def __init__(self, table, returning=None, **kwargs):
        self.table = table
        self._returning = returning
        self._return_cursor = True if returning else False
        super(_WriteQuery, self).__init__(**kwargs)

    def cte(self, name, recursive=False, columns=None, materialized=None):
        return CTE(name, self, recursive=recursive, columns=columns, materialized=materialized)

    @Node.copy
    def returning(self, *returning):
        self._returning = returning
        self._return_cursor = True if returning else False

    def apply_returning(self, ctx):
        if self._returning:
            with ctx.scope_source():
                ctx.literal(' RETURNING ').sql(CommaNodeList(self._returning))
        return ctx

    def _execute(self, database):
        if self._returning:
            cursor = self.execute_returning(database)
        else:
            cursor = database.execute(self)
        return self.handle_result(database, cursor)

    def execute_returning(self, database):
        if self._cursor_wrapper is None:
            cursor = database.execute(self)
            self._cursor_wrapper = self._get_cursor_wrapper(cursor)
        return self._cursor_wrapper

    def handle_result(self, database, cursor):
        if self._return_cursor:
            return cursor
        return database.rows_affected(cursor)

    def _set_table_alias(self, ctx):
        ctx.alias_manager[self.table] = self.table.__name__

    def __sql__(self, ctx):
        super(_WriteQuery, self).__sql__(ctx)
        self._set_table_alias(ctx)
        return ctx