import os
import sys
import subprocess
from urllib.parse import quote
from paste.util import converters
class CGIWriter(object):

    def __init__(self, environ, start_response):
        self.environ = environ
        self.start_response = start_response
        self.status = '200 OK'
        self.headers = []
        self.headers_finished = False
        self.writer = None
        self.buffer = b''

    def write(self, data):
        if self.headers_finished:
            self.writer(data)
            return
        self.buffer += data
        while b'\n' in self.buffer:
            if b'\r\n' in self.buffer and self.buffer.find(b'\r\n') < self.buffer.find(b'\n'):
                line1, self.buffer = self.buffer.split(b'\r\n', 1)
            else:
                line1, self.buffer = self.buffer.split(b'\n', 1)
            if not line1:
                self.headers_finished = True
                self.writer = self.start_response(self.status, self.headers)
                self.writer(self.buffer)
                del self.buffer
                del self.headers
                del self.status
                break
            elif b':' not in line1:
                raise CGIError('Bad header line: %r' % line1)
            else:
                name, value = line1.split(b':', 1)
                value = value.lstrip()
                name = name.strip()
                name = name.decode('utf8')
                value = value.decode('utf8')
                if name.lower() == 'status':
                    if ' ' not in value:
                        value = '%s General' % value
                    self.status = value
                else:
                    self.headers.append((name, value))