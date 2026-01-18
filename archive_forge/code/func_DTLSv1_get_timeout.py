import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def DTLSv1_get_timeout(self):
    """
        Determine when the DTLS SSL object next needs to perform internal
        processing due to the passage of time.

        When the returned number of seconds have passed, the
        :meth:`DTLSv1_handle_timeout` method needs to be called.

        :return: The time left in seconds before the next timeout or `None`
            if no timeout is currently active.
        """
    ptv_sec = _ffi.new('time_t *')
    ptv_usec = _ffi.new('long *')
    if _lib.Cryptography_DTLSv1_get_timeout(self._ssl, ptv_sec, ptv_usec):
        return ptv_sec[0] + ptv_usec[0] / 1000000
    else:
        return None