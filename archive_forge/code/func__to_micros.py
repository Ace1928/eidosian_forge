from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
def _to_micros(dur: timedelta) -> int:
    """Convert duration 'dur' to microseconds."""
    return int(dur.total_seconds() * 1000000.0)