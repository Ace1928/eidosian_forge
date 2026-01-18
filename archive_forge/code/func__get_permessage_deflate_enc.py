import base64
import codecs
import collections
import errno
from random import Random
from socket import error as SocketError
import string
import struct
import sys
import time
import zlib
from eventlet import semaphore
from eventlet import wsgi
from eventlet.green import socket
from eventlet.support import get_errno
def _get_permessage_deflate_enc(self):
    options = self.extensions.get('permessage-deflate')
    if options is None:
        return None

    def _make():
        return zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -options.get('client_max_window_bits' if self.client else 'server_max_window_bits', zlib.MAX_WBITS))
    if options.get('client_no_context_takeover' if self.client else 'server_no_context_takeover'):
        return _make()
    else:
        if self._deflate_enc is None:
            self._deflate_enc = _make()
        return self._deflate_enc