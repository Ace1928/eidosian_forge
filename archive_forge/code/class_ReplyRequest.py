from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class ReplyRequest(GetAttrData):

    def __init__(self, display, defer=0, *args, **keys):
        self._display = display
        self._binary = self._request.to_binary(*args, **keys)
        self._serial = None
        self._data = None
        self._error = None
        self._response_lock = lock.allocate_lock()
        self._display.send_request(self, 1)
        if not defer:
            self.reply()

    def reply(self):
        self._response_lock.acquire()
        while self._data is None and self._error is None:
            self._display.send_recv_lock.acquire()
            self._response_lock.release()
            self._display.send_and_recv(request=self._serial)
            self._response_lock.acquire()
        self._response_lock.release()
        self._display = None
        if self._error:
            raise self._error

    def _parse_response(self, data):
        self._response_lock.acquire()
        self._data, d = self._reply.parse_binary(data, self._display, rawdict=1)
        self._response_lock.release()

    def _set_error(self, error):
        self._response_lock.acquire()
        self._error = error
        self._response_lock.release()
        return 1

    def __repr__(self):
        return '<%s serial = %s, data = %s, error = %s>' % (self.__class__, self._serial, self._data, self._error)