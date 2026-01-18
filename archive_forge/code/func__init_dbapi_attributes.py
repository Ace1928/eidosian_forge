import asyncio
from functools import partial
from .base import SQLiteExecutionContext
from .pysqlite import SQLiteDialect_pysqlite
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
def _init_dbapi_attributes(self):
    for name in ('DatabaseError', 'Error', 'IntegrityError', 'NotSupportedError', 'OperationalError', 'ProgrammingError', 'sqlite_version', 'sqlite_version_info'):
        setattr(self, name, getattr(self.aiosqlite, name))
    for name in ('PARSE_COLNAMES', 'PARSE_DECLTYPES'):
        setattr(self, name, getattr(self.sqlite, name))
    for name in ('Binary',):
        setattr(self, name, getattr(self.sqlite, name))