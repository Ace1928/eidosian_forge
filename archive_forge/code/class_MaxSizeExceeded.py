import errno
import sys
class MaxSizeExceeded(Exception):
    """Exception raised when a client sends more data then allowed under limit.

    Depends on ``request.body.maxbytes`` config option if used within CherryPy.
    """