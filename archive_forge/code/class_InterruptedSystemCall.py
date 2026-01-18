from __future__ import annotations
from errno import EINTR
class InterruptedSystemCall(ZMQError, InterruptedError):
    """Wrapper for EINTR

    This exception should be caught internally in pyzmq
    to retry system calls, and not propagate to the user.

    .. versionadded:: 14.7
    """
    errno = EINTR

    def __init__(self, errno='ignored', msg='ignored'):
        super().__init__(EINTR)

    def __str__(self):
        s = super().__str__()
        return s + ': This call should have been retried. Please report this to pyzmq.'