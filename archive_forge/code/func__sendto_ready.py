import collections
import errno
import functools
import selectors
import socket
import warnings
import weakref
from . import base_events
from . import constants
from . import events
from . import futures
from . import protocols
from . import sslproto
from . import transports
from . import trsock
from .log import logger
def _sendto_ready(self):
    while self._buffer:
        data, addr = self._buffer.popleft()
        self._buffer_size -= len(data)
        try:
            if self._extra['peername']:
                self._sock.send(data)
            else:
                self._sock.sendto(data, addr)
        except (BlockingIOError, InterruptedError):
            self._buffer.appendleft((data, addr))
            self._buffer_size += len(data)
            break
        except OSError as exc:
            self._protocol.error_received(exc)
            return
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            self._fatal_error(exc, 'Fatal write error on datagram transport')
            return
    self._maybe_resume_protocol()
    if not self._buffer:
        self._loop._remove_writer(self._sock_fd)
        if self._closing:
            self._call_connection_lost(None)