import io
import os
import sys
import socket
import struct
import time
import tempfile
import itertools
from . import util
from . import AuthenticationError, BufferTooShort
from .context import reduction
class Listener(object):
    """
    Returns a listener object.

    This is a wrapper for a bound socket which is 'listening' for
    connections, or for a Windows named pipe.
    """

    def __init__(self, address=None, family=None, backlog=1, authkey=None):
        family = family or (address and address_type(address)) or default_family
        address = address or arbitrary_address(family)
        _validate_family(family)
        if family == 'AF_PIPE':
            self._listener = PipeListener(address, backlog)
        else:
            self._listener = SocketListener(address, family, backlog)
        if authkey is not None and (not isinstance(authkey, bytes)):
            raise TypeError('authkey should be a byte string')
        self._authkey = authkey

    def accept(self):
        """
        Accept a connection on the bound socket or named pipe of `self`.

        Returns a `Connection` object.
        """
        if self._listener is None:
            raise OSError('listener is closed')
        c = self._listener.accept()
        if self._authkey:
            deliver_challenge(c, self._authkey)
            answer_challenge(c, self._authkey)
        return c

    def close(self):
        """
        Close the bound socket or named pipe of `self`.
        """
        listener = self._listener
        if listener is not None:
            self._listener = None
            listener.close()

    @property
    def address(self):
        return self._listener._address

    @property
    def last_accepted(self):
        return self._listener._last_accepted

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()