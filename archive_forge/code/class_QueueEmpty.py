import collections
import heapq
from types import GenericAlias
from . import locks
from . import mixins
class QueueEmpty(Exception):
    """Raised when Queue.get_nowait() is called on an empty Queue."""
    pass