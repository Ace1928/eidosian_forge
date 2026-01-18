from __future__ import annotations
import calendar
import datetime
import functools
from typing import Any, Union, cast
from bson.codec_options import DEFAULT_CODEC_OPTIONS, CodecOptions, DatetimeConversion
from bson.errors import InvalidBSON
from bson.tz_util import utc
def _millis_to_datetime(millis: int, opts: CodecOptions[Any]) -> Union[datetime.datetime, DatetimeMS]:
    """Convert milliseconds since epoch UTC to datetime."""
    if opts.datetime_conversion == DatetimeConversion.DATETIME or opts.datetime_conversion == DatetimeConversion.DATETIME_CLAMP or opts.datetime_conversion == DatetimeConversion.DATETIME_AUTO:
        tz = opts.tzinfo or datetime.timezone.utc
        if opts.datetime_conversion == DatetimeConversion.DATETIME_CLAMP:
            millis = max(_min_datetime_ms(tz), min(millis, _max_datetime_ms(tz)))
        elif opts.datetime_conversion == DatetimeConversion.DATETIME_AUTO:
            if not _min_datetime_ms(tz) <= millis <= _max_datetime_ms(tz):
                return DatetimeMS(millis)
        diff = (millis % 1000 + 1000) % 1000
        seconds = (millis - diff) // 1000
        micros = diff * 1000
        try:
            if opts.tz_aware:
                dt = EPOCH_AWARE + datetime.timedelta(seconds=seconds, microseconds=micros)
                if opts.tzinfo:
                    dt = dt.astimezone(tz)
                return dt
            else:
                return EPOCH_NAIVE + datetime.timedelta(seconds=seconds, microseconds=micros)
        except ArithmeticError as err:
            raise InvalidBSON(f'{err} {_DATETIME_ERROR_SUGGESTION}') from err
    elif opts.datetime_conversion == DatetimeConversion.DATETIME_MS:
        return DatetimeMS(millis)
    else:
        raise ValueError('datetime_conversion must be an element of DatetimeConversion')