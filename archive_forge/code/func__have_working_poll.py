import errno
import select
import sys
from functools import partial
def _have_working_poll():
    try:
        poll_obj = select.poll()
        _retry_on_intr(poll_obj.poll, 0)
    except (AttributeError, OSError):
        return False
    else:
        return True