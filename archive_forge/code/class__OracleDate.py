from __future__ import annotations
import datetime as dt
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING
from ... import exc
from ...sql import sqltypes
from ...types import NVARCHAR
from ...types import VARCHAR
class _OracleDate(_OracleDateLiteralRender, sqltypes.Date):

    def literal_processor(self, dialect):
        return self._literal_processor_date(dialect)