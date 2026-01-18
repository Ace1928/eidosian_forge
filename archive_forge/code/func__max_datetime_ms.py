from __future__ import annotations
import calendar
import datetime
import functools
from typing import Any, Union, cast
from bson.codec_options import DEFAULT_CODEC_OPTIONS, CodecOptions, DatetimeConversion
from bson.errors import InvalidBSON
from bson.tz_util import utc
@functools.lru_cache(maxsize=None)
def _max_datetime_ms(tz: datetime.timezone=datetime.timezone.utc) -> int:
    return _datetime_to_millis(datetime.datetime.max.replace(tzinfo=tz))