import collections
from contextlib import contextmanager
from eventlet import queue
class TokenPool(Pool):
    """A pool which gives out tokens (opaque unique objects), which indicate
    that the coroutine which holds the token has a right to consume some
    limited resource.
    """

    def create(self):
        return Token()