from __future__ import annotations
from errno import EINTR
class NotDone(ZMQBaseError):
    """Raised when timeout is reached while waiting for 0MQ to finish with a Message

    See Also
    --------
    .MessageTracker.wait : object for tracking when ZeroMQ is done
    """