import errno
import select
import sys
from functools import partial
def _retry_on_intr(fn, timeout):
    if timeout is None:
        deadline = float('inf')
    else:
        deadline = monotonic() + timeout
    while True:
        try:
            return fn(timeout)
        except (OSError, select.error) as e:
            if e.args[0] != errno.EINTR:
                raise
            else:
                timeout = deadline - monotonic()
                if timeout < 0:
                    timeout = 0
                if timeout == float('inf'):
                    timeout = None
                continue