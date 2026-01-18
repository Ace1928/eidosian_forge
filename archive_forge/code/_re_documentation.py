from __future__ import annotations
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from functools import lru_cache
import re
from typing import Any
from ._types import ParseFloat
Convert a `RE_DATETIME` match to `datetime.datetime` or `datetime.date`.

    Raises ValueError if the match does not correspond to a valid date
    or datetime.
    