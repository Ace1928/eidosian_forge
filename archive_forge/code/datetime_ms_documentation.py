from __future__ import annotations
import calendar
import datetime
import functools
from typing import Any, Union, cast
from bson.codec_options import DEFAULT_CODEC_OPTIONS, CodecOptions, DatetimeConversion
from bson.errors import InvalidBSON
from bson.tz_util import utc
Create a Python :class:`~datetime.datetime` from this DatetimeMS object.

        :Parameters:
          - `codec_options`: A CodecOptions instance for specifying how the
            resulting DatetimeMS object will be formatted using ``tz_aware``
            and ``tz_info``. Defaults to
            :const:`~bson.codec_options.DEFAULT_CODEC_OPTIONS`.
        