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
def _parse_extension_header(self, header):
    if header is None:
        return None
    res = {}
    for ext in header.split(','):
        parts = ext.split(';')
        config = {}
        for part in parts[1:]:
            key_val = part.split('=')
            if len(key_val) == 1:
                config[key_val[0].strip().lower()] = True
            else:
                config[key_val[0].strip().lower()] = key_val[1].strip().strip('"').lower()
        res.setdefault(parts[0].strip().lower(), []).append(config)
    return res