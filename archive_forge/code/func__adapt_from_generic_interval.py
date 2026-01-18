from __future__ import annotations
import datetime as dt
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING
from ... import exc
from ...sql import sqltypes
from ...types import NVARCHAR
from ...types import VARCHAR
@classmethod
def _adapt_from_generic_interval(cls, interval):
    return INTERVAL(day_precision=interval.day_precision, second_precision=interval.second_precision)