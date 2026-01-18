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
class UTF8Decoder:

    def __init__(self):
        if utf8validator:
            self.validator = utf8validator.Utf8Validator()
        else:
            self.validator = None
        decoderclass = codecs.getincrementaldecoder('utf8')
        self.decoder = decoderclass()

    def reset(self):
        if self.validator:
            self.validator.reset()
        self.decoder.reset()

    def decode(self, data, final=False):
        if self.validator:
            valid, eocp, c_i, t_i = self.validator.validate(data)
            if not valid:
                raise ValueError('Data is not valid unicode')
        return self.decoder.decode(data, final)