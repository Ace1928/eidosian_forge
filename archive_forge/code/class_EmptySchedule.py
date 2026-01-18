from __future__ import annotations
from heapq import heappop, heappush
from itertools import count
from types import MethodType
from typing import (
from simpy.events import (
class EmptySchedule(Exception):
    """Thrown by an :class:`Environment` if there are no further events to be
    processed."""