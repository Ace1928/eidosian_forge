import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def _get_finished_message(self, function):
    """
        Helper to implement :meth:`get_finished` and
        :meth:`get_peer_finished`.

        :param function: Either :data:`SSL_get_finished`: or
            :data:`SSL_get_peer_finished`.

        :return: :data:`None` if the desired message has not yet been
            received, otherwise the contents of the message.
        :rtype: :class:`bytes` or :class:`NoneType`
        """
    empty = _ffi.new('char[]', 0)
    size = function(self._ssl, empty, 0)
    if size == 0:
        return None
    buf = _no_zero_allocator('char[]', size)
    function(self._ssl, buf, size)
    return _ffi.buffer(buf, size)[:]