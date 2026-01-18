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
def _extract_number(self, value):
    """
        Utility function which, given a string like 'g98sd  5[]221@1', will
        return 9852211. Used to parse the Sec-WebSocket-Key headers.
        """
    out = ''
    spaces = 0
    for char in value:
        if char in string.digits:
            out += char
        elif char == ' ':
            spaces += 1
    return int(out) // spaces