from base64 import b64encode
import six
from errno import EOPNOTSUPP, EINVAL, EAGAIN
import functools
from io import BytesIO
import logging
import os
from os import SEEK_CUR
import socket
import struct
import sys
class _BaseSocket(socket.socket):
    """Allows Python 2 delegated methods such as send() to be overridden."""

    def __init__(self, *pos, **kw):
        _orig_socket.__init__(self, *pos, **kw)
        self._savedmethods = dict()
        for name in self._savenames:
            self._savedmethods[name] = getattr(self, name)
            delattr(self, name)
    _savenames = list()