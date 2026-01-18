import collections
import sys
import warnings
from . import protocols
from . import transports
from .log import logger
def _process_write_backlog(self):
    if self._transport is None:
        return
    try:
        for i in range(len(self._write_backlog)):
            data, offset = self._write_backlog[0]
            if data:
                ssldata, offset = self._sslpipe.feed_appdata(data, offset)
            elif offset:
                ssldata = self._sslpipe.do_handshake(self._on_handshake_complete)
                offset = 1
            else:
                ssldata = self._sslpipe.shutdown(self._finalize)
                offset = 1
            for chunk in ssldata:
                self._transport.write(chunk)
            if offset < len(data):
                self._write_backlog[0] = (data, offset)
                assert self._sslpipe.need_ssldata
                if self._transport._paused:
                    self._transport.resume_reading()
                break
            del self._write_backlog[0]
            self._write_buffer_size -= len(data)
    except BaseException as exc:
        if self._in_handshake:
            self._on_handshake_complete(exc)
        else:
            self._fatal_error(exc, 'Fatal error on SSL transport')