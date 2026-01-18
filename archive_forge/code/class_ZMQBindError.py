from __future__ import annotations
from errno import EINTR
class ZMQBindError(ZMQBaseError):
    """An error for ``Socket.bind_to_random_port()``.

    See Also
    --------
    .Socket.bind_to_random_port
    """