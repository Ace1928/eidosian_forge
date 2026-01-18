import json
import math
import re
import struct
import sys
from peewee import *
from peewee import ColumnBase
from peewee import EnclosedNodeList
from peewee import Entity
from peewee import Expression
from peewee import Insert
from peewee import Node
from peewee import NodeList
from peewee import OP
from peewee import VirtualField
from peewee import merge_dict
from peewee import sqlite3
class CSqliteExtDatabase(SqliteExtDatabase):

    def __init__(self, *args, **kwargs):
        self._conn_helper = None
        self._commit_hook = self._rollback_hook = self._update_hook = None
        self._replace_busy_handler = False
        super(CSqliteExtDatabase, self).__init__(*args, **kwargs)

    def init(self, database, replace_busy_handler=False, **kwargs):
        super(CSqliteExtDatabase, self).init(database, **kwargs)
        self._replace_busy_handler = replace_busy_handler

    def _close(self, conn):
        if self._commit_hook:
            self._conn_helper.set_commit_hook(None)
        if self._rollback_hook:
            self._conn_helper.set_rollback_hook(None)
        if self._update_hook:
            self._conn_helper.set_update_hook(None)
        return super(CSqliteExtDatabase, self)._close(conn)

    def _add_conn_hooks(self, conn):
        super(CSqliteExtDatabase, self)._add_conn_hooks(conn)
        self._conn_helper = ConnectionHelper(conn)
        if self._commit_hook is not None:
            self._conn_helper.set_commit_hook(self._commit_hook)
        if self._rollback_hook is not None:
            self._conn_helper.set_rollback_hook(self._rollback_hook)
        if self._update_hook is not None:
            self._conn_helper.set_update_hook(self._update_hook)
        if self._replace_busy_handler:
            timeout = self._timeout or 5
            self._conn_helper.set_busy_handler(timeout * 1000)

    def on_commit(self, fn):
        self._commit_hook = fn
        if not self.is_closed():
            self._conn_helper.set_commit_hook(fn)
        return fn

    def on_rollback(self, fn):
        self._rollback_hook = fn
        if not self.is_closed():
            self._conn_helper.set_rollback_hook(fn)
        return fn

    def on_update(self, fn):
        self._update_hook = fn
        if not self.is_closed():
            self._conn_helper.set_update_hook(fn)
        return fn

    def changes(self):
        return self._conn_helper.changes()

    @property
    def last_insert_rowid(self):
        return self._conn_helper.last_insert_rowid()

    @property
    def autocommit(self):
        return self._conn_helper.autocommit()

    def backup(self, destination, pages=None, name=None, progress=None):
        return backup(self.connection(), destination.connection(), pages=pages, name=name, progress=progress)

    def backup_to_file(self, filename, pages=None, name=None, progress=None):
        return backup_to_file(self.connection(), filename, pages=pages, name=name, progress=progress)

    def blob_open(self, table, column, rowid, read_only=False):
        return Blob(self, table, column, rowid, read_only)
    memory_used = __status__(SQLITE_STATUS_MEMORY_USED)
    malloc_size = __status__(SQLITE_STATUS_MALLOC_SIZE, True)
    malloc_count = __status__(SQLITE_STATUS_MALLOC_COUNT)
    pagecache_used = __status__(SQLITE_STATUS_PAGECACHE_USED)
    pagecache_overflow = __status__(SQLITE_STATUS_PAGECACHE_OVERFLOW)
    pagecache_size = __status__(SQLITE_STATUS_PAGECACHE_SIZE, True)
    scratch_used = __status__(SQLITE_STATUS_SCRATCH_USED)
    scratch_overflow = __status__(SQLITE_STATUS_SCRATCH_OVERFLOW)
    scratch_size = __status__(SQLITE_STATUS_SCRATCH_SIZE, True)
    lookaside_used = __dbstatus__(SQLITE_DBSTATUS_LOOKASIDE_USED)
    lookaside_hit = __dbstatus__(SQLITE_DBSTATUS_LOOKASIDE_HIT, True)
    lookaside_miss = __dbstatus__(SQLITE_DBSTATUS_LOOKASIDE_MISS_SIZE, True)
    lookaside_miss_full = __dbstatus__(SQLITE_DBSTATUS_LOOKASIDE_MISS_FULL, True)
    cache_used = __dbstatus__(SQLITE_DBSTATUS_CACHE_USED, False, True)
    schema_used = __dbstatus__(SQLITE_DBSTATUS_SCHEMA_USED, False, True)
    statement_used = __dbstatus__(SQLITE_DBSTATUS_STMT_USED, False, True)
    cache_hit = __dbstatus__(SQLITE_DBSTATUS_CACHE_HIT, False, True)
    cache_miss = __dbstatus__(SQLITE_DBSTATUS_CACHE_MISS, False, True)
    cache_write = __dbstatus__(SQLITE_DBSTATUS_CACHE_WRITE, False, True)