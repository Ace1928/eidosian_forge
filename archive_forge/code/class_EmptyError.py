import threading
from collections import deque
from time import time
from sentry_sdk._types import TYPE_CHECKING
class EmptyError(Exception):
    """Exception raised by Queue.get(block=0)/get_nowait()."""
    pass