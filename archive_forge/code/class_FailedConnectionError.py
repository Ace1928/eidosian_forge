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
class FailedConnectionError(Exception):

    def __init__(self, status, message):
        super().__init__(status, message)
        self.message = message
        self.status = status