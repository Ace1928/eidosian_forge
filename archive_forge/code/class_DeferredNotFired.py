from typing import List, TYPE_CHECKING
from functools import partial
from testtools.content import TracebackContent
class DeferredNotFired(Exception):
    """Raised when we extract a result from a Deferred that's not fired yet."""

    def __init__(self, deferred):
        msg = f'{deferred!r} has not fired yet.'
        super().__init__(msg)