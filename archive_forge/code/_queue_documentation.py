import threading
from collections import deque
from time import time
from sentry_sdk._types import TYPE_CHECKING
Remove and return an item from the queue without blocking.

        Only get an item if one is immediately available. Otherwise
        raise the EmptyError exception.
        