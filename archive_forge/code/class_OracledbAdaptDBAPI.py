from __future__ import annotations
import collections
import re
from typing import Any
from typing import TYPE_CHECKING
from .cx_oracle import OracleDialect_cx_oracle as _OracleDialect_cx_oracle
from ... import exc
from ... import pool
from ...connectors.asyncio import AsyncAdapt_dbapi_connection
from ...connectors.asyncio import AsyncAdapt_dbapi_cursor
from ...connectors.asyncio import AsyncAdaptFallback_dbapi_connection
from ...util import asbool
from ...util import await_fallback
from ...util import await_only
class OracledbAdaptDBAPI:

    def __init__(self, oracledb) -> None:
        self.oracledb = oracledb
        for k, v in self.oracledb.__dict__.items():
            if k != 'connect':
                self.__dict__[k] = v

    def connect(self, *arg, **kw):
        async_fallback = kw.pop('async_fallback', False)
        creator_fn = kw.pop('async_creator_fn', self.oracledb.connect_async)
        if asbool(async_fallback):
            return AsyncAdaptFallback_oracledb_connection(self, await_fallback(creator_fn(*arg, **kw)))
        else:
            return AsyncAdapt_oracledb_connection(self, await_only(creator_fn(*arg, **kw)))