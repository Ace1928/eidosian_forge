from __future__ import annotations
from errno import EINTR
class Again(ZMQError):
    """Wrapper for zmq.EAGAIN

    .. versionadded:: 13.0
    """

    def __init__(self, errno='ignored', msg='ignored'):
        from zmq import EAGAIN
        super().__init__(EAGAIN)