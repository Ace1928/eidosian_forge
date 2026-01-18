from __future__ import annotations
import re
from bisect import bisect, bisect_left
from collections import namedtuple
from collections.abc import Iterable
from datetime import datetime, timedelta, tzinfo
from typing import Any, Callable, Mapping, Sequence
from kombu.utils.objects import cached_property
from celery import Celery
from . import current_app
from .utils.collections import AttributeDict
from .utils.time import (ffwd, humanize_seconds, localize, maybe_make_aware, maybe_timedelta, remaining, timezone,
def _range_steps(self, toks: Sequence[str]) -> list[int]:
    if len(toks) != 3 or not toks[2]:
        raise self.ParseException('empty filter')
    return self._expand_range(toks[:2])[::int(toks[2])]