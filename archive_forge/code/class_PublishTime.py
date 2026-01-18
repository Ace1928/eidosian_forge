import enum
from datetime import datetime
from typing import NamedTuple
class PublishTime(NamedTuple):
    """The publish timestamp of a message."""
    value: datetime