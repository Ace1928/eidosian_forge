from __future__ import annotations
import datetime as dt
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING
from ... import exc
from ...sql import sqltypes
from ...types import NVARCHAR
from ...types import VARCHAR
class _OracleBoolean(sqltypes.Boolean):

    def get_dbapi_type(self, dbapi):
        return dbapi.NUMBER