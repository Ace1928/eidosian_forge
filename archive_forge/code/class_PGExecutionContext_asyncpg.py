from __future__ import annotations
import collections
import decimal
import json as _py_json
import re
import time
from . import json
from . import ranges
from .array import ARRAY as PGARRAY
from .base import _DECIMAL_TYPES
from .base import _FLOAT_TYPES
from .base import _INT_TYPES
from .base import ENUM
from .base import INTERVAL
from .base import OID
from .base import PGCompiler
from .base import PGDialect
from .base import PGExecutionContext
from .base import PGIdentifierPreparer
from .base import REGCLASS
from .base import REGCONFIG
from .types import BIT
from .types import BYTEA
from .types import CITEXT
from ... import exc
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...engine import processors
from ...sql import sqltypes
from ...util.concurrency import asyncio
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class PGExecutionContext_asyncpg(PGExecutionContext):

    def handle_dbapi_exception(self, e):
        if isinstance(e, (self.dialect.dbapi.InvalidCachedStatementError, self.dialect.dbapi.InternalServerError)):
            self.dialect._invalidate_schema_cache()

    def pre_exec(self):
        if self.isddl:
            self.dialect._invalidate_schema_cache()
        self.cursor._invalidate_schema_cache_asof = self.dialect._invalidate_schema_cache_asof
        if not self.compiled:
            return

    def create_server_side_cursor(self):
        return self._dbapi_connection.cursor(server_side=True)