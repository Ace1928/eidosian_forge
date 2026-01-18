from __future__ import annotations
import datetime as dt
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING
from ... import exc
from ...sql import sqltypes
from ...types import NVARCHAR
from ...types import VARCHAR
def _literal_processor_date(self, dialect):

    def process(value):
        if getattr(value, 'microsecond', None):
            value = f"TO_TIMESTAMP('{value.isoformat().split('T')[0]}', 'YYYY-MM-DD')"
        else:
            value = f"TO_DATE('{value.isoformat().split('T')[0]}', 'YYYY-MM-DD')"
        return value
    return process