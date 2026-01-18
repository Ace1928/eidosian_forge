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
class OracleDialect_oracledb(_OracleDialect_cx_oracle):
    supports_statement_cache = True
    driver = 'oracledb'
    _min_version = (1,)

    def __init__(self, auto_convert_lobs=True, coerce_to_decimal=True, arraysize=None, encoding_errors=None, thick_mode=None, **kwargs):
        super().__init__(auto_convert_lobs, coerce_to_decimal, arraysize, encoding_errors, **kwargs)
        if self.dbapi is not None and (thick_mode or isinstance(thick_mode, dict)):
            kw = thick_mode if isinstance(thick_mode, dict) else {}
            self.dbapi.init_oracle_client(**kw)

    @classmethod
    def import_dbapi(cls):
        import oracledb
        return oracledb

    @classmethod
    def is_thin_mode(cls, connection):
        return connection.connection.dbapi_connection.thin

    @classmethod
    def get_async_dialect_cls(cls, url):
        return OracleDialectAsync_oracledb

    def _load_version(self, dbapi_module):
        version = (0, 0, 0)
        if dbapi_module is not None:
            m = re.match('(\\d+)\\.(\\d+)(?:\\.(\\d+))?', dbapi_module.version)
            if m:
                version = tuple((int(x) for x in m.group(1, 2, 3) if x is not None))
        self.oracledb_ver = version
        if self.oracledb_ver > (0, 0, 0) and self.oracledb_ver < self._min_version:
            raise exc.InvalidRequestError(f'oracledb version {self._min_version} and above are supported')