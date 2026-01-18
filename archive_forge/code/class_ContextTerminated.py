from __future__ import annotations
from errno import EINTR
class ContextTerminated(ZMQError):
    """Wrapper for zmq.ETERM

    .. versionadded:: 13.0
    """

    def __init__(self, errno='ignored', msg='ignored'):
        from zmq import ETERM
        super().__init__(ETERM)