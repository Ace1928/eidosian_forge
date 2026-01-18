import io
import logging
import os
import re
import sys
from gunicorn.http.message import HEADER_RE
from gunicorn.http.errors import InvalidHeader, InvalidHeaderName
from gunicorn import SERVER_SOFTWARE, SERVER
from gunicorn import util
class WSGIErrorsWrapper(io.RawIOBase):

    def __init__(self, cfg):
        errorlog = logging.getLogger('gunicorn.error')
        handlers = errorlog.handlers
        self.streams = []
        if cfg.errorlog == '-':
            self.streams.append(sys.stderr)
            handlers = handlers[1:]
        for h in handlers:
            if hasattr(h, 'stream'):
                self.streams.append(h.stream)

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
            except UnicodeError:
                stream.write(data.encode('UTF-8'))
            stream.flush()