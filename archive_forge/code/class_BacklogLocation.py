import enum
from datetime import datetime
from typing import NamedTuple
class BacklogLocation(enum.Enum):
    """A location with respect to the message backlog. BEGINNING refers to the
    location of the oldest retained message. END refers to the location past
    all currently published messages, skipping the entire message backlog."""
    BEGINNING = 0
    END = 1