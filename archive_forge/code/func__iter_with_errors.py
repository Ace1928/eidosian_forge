import _thread
import struct
import sys
import time
from collections import deque
from io import BytesIO
from fastbencode import bdecode_as_tuple, bencode
import breezy
from ... import debug, errors, osutils
from ...trace import log_exception_quietly, mutter
from . import message, request
def _iter_with_errors(iterable):
    """Handle errors from iterable.next().

    Use like::

        for exc_info, value in _iter_with_errors(iterable):
            ...

    This is a safer alternative to::

        try:
            for value in iterable:
               ...
        except:
            ...

    Because the latter will catch errors from the for-loop body, not just
    iterable.next()

    If an error occurs, exc_info will be a exc_info tuple, and the generator
    will terminate.  Otherwise exc_info will be None, and value will be the
    value from iterable.next().  Note that KeyboardInterrupt and SystemExit
    will not be itercepted.
    """
    iterator = iter(iterable)
    while True:
        try:
            yield (None, next(iterator))
        except StopIteration:
            return
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            mutter('_iter_with_errors caught error')
            log_exception_quietly()
            yield (sys.exc_info(), None)
            return