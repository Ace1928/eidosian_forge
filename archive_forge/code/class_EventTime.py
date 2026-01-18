import enum
from datetime import datetime
from typing import NamedTuple
class EventTime(NamedTuple):
    """A user-defined event timestamp of a message."""
    value: datetime