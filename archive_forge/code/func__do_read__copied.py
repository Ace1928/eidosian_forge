import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _do_read__copied(self):
    chunk = b'1'
    zero = True
    one = False
    try:
        while True:
            chunk = self._sslobj.read(self.max_size)
            if not chunk:
                break
            if zero:
                zero = False
                one = True
                first = chunk
            elif one:
                one = False
                data = [first, chunk]
            else:
                data.append(chunk)
    except SSLAgainErrors:
        pass
    if one:
        self._app_protocol.data_received(first)
    elif not zero:
        self._app_protocol.data_received(b''.join(data))
    if not chunk:
        self._call_eof_received()
        self._start_shutdown()