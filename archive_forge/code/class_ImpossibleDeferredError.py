from typing import List, TYPE_CHECKING
from functools import partial
from testtools.content import TracebackContent
class ImpossibleDeferredError(Exception):
    """Raised if a Deferred somehow triggers both a success and a failure."""

    def __init__(self, deferred, successes, failures):
        msg = 'Impossible condition on %r, got both success (%r) and failure (%r)'
        super().__init__(msg % (deferred, successes, failures))