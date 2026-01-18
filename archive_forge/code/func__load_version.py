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
def _load_version(self, dbapi_module):
    version = (0, 0, 0)
    if dbapi_module is not None:
        m = re.match('(\\d+)\\.(\\d+)(?:\\.(\\d+))?', dbapi_module.version)
        if m:
            version = tuple((int(x) for x in m.group(1, 2, 3) if x is not None))
    self.oracledb_ver = version
    if self.oracledb_ver > (0, 0, 0) and self.oracledb_ver < self._min_version:
        raise exc.InvalidRequestError(f'oracledb version {self._min_version} and above are supported')