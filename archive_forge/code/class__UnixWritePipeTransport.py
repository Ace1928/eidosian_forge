import errno
import io
import itertools
import os
import selectors
import signal
import socket
import stat
import subprocess
import sys
import threading
import warnings
from . import base_events
from . import base_subprocess
from . import constants
from . import coroutines
from . import events
from . import exceptions
from . import futures
from . import selector_events
from . import tasks
from . import transports
from .log import logger
class _UnixWritePipeTransport(transports._FlowControlMixin, transports.WriteTransport):

    def __init__(self, loop, pipe, protocol, waiter=None, extra=None):
        super().__init__(extra, loop)
        self._extra['pipe'] = pipe
        self._pipe = pipe
        self._fileno = pipe.fileno()
        self._protocol = protocol
        self._buffer = bytearray()
        self._conn_lost = 0
        self._closing = False
        mode = os.fstat(self._fileno).st_mode
        is_char = stat.S_ISCHR(mode)
        is_fifo = stat.S_ISFIFO(mode)
        is_socket = stat.S_ISSOCK(mode)
        if not (is_char or is_fifo or is_socket):
            self._pipe = None
            self._fileno = None
            self._protocol = None
            raise ValueError('Pipe transport is only for pipes, sockets and character devices')
        os.set_blocking(self._fileno, False)
        self._loop.call_soon(self._protocol.connection_made, self)
        if is_socket or (is_fifo and (not sys.platform.startswith('aix'))):
            self._loop.call_soon(self._loop._add_reader, self._fileno, self._read_ready)
        if waiter is not None:
            self._loop.call_soon(futures._set_result_unless_cancelled, waiter, None)

    def __repr__(self):
        info = [self.__class__.__name__]
        if self._pipe is None:
            info.append('closed')
        elif self._closing:
            info.append('closing')
        info.append(f'fd={self._fileno}')
        selector = getattr(self._loop, '_selector', None)
        if self._pipe is not None and selector is not None:
            polling = selector_events._test_selector_event(selector, self._fileno, selectors.EVENT_WRITE)
            if polling:
                info.append('polling')
            else:
                info.append('idle')
            bufsize = self.get_write_buffer_size()
            info.append(f'bufsize={bufsize}')
        elif self._pipe is not None:
            info.append('open')
        else:
            info.append('closed')
        return '<{}>'.format(' '.join(info))

    def get_write_buffer_size(self):
        return len(self._buffer)

    def _read_ready(self):
        if self._loop.get_debug():
            logger.info('%r was closed by peer', self)
        if self._buffer:
            self._close(BrokenPipeError())
        else:
            self._close()

    def write(self, data):
        assert isinstance(data, (bytes, bytearray, memoryview)), repr(data)
        if isinstance(data, bytearray):
            data = memoryview(data)
        if not data:
            return
        if self._conn_lost or self._closing:
            if self._conn_lost >= constants.LOG_THRESHOLD_FOR_CONNLOST_WRITES:
                logger.warning('pipe closed by peer or os.write(pipe, data) raised exception.')
            self._conn_lost += 1
            return
        if not self._buffer:
            try:
                n = os.write(self._fileno, data)
            except (BlockingIOError, InterruptedError):
                n = 0
            except (SystemExit, KeyboardInterrupt):
                raise
            except BaseException as exc:
                self._conn_lost += 1
                self._fatal_error(exc, 'Fatal write error on pipe transport')
                return
            if n == len(data):
                return
            elif n > 0:
                data = memoryview(data)[n:]
            self._loop._add_writer(self._fileno, self._write_ready)
        self._buffer += data
        self._maybe_pause_protocol()

    def _write_ready(self):
        assert self._buffer, 'Data should not be empty'
        try:
            n = os.write(self._fileno, self._buffer)
        except (BlockingIOError, InterruptedError):
            pass
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            self._buffer.clear()
            self._conn_lost += 1
            self._loop._remove_writer(self._fileno)
            self._fatal_error(exc, 'Fatal write error on pipe transport')
        else:
            if n == len(self._buffer):
                self._buffer.clear()
                self._loop._remove_writer(self._fileno)
                self._maybe_resume_protocol()
                if self._closing:
                    self._loop._remove_reader(self._fileno)
                    self._call_connection_lost(None)
                return
            elif n > 0:
                del self._buffer[:n]

    def can_write_eof(self):
        return True

    def write_eof(self):
        if self._closing:
            return
        assert self._pipe
        self._closing = True
        if not self._buffer:
            self._loop._remove_reader(self._fileno)
            self._loop.call_soon(self._call_connection_lost, None)

    def set_protocol(self, protocol):
        self._protocol = protocol

    def get_protocol(self):
        return self._protocol

    def is_closing(self):
        return self._closing

    def close(self):
        if self._pipe is not None and (not self._closing):
            self.write_eof()

    def __del__(self, _warn=warnings.warn):
        if self._pipe is not None:
            _warn(f'unclosed transport {self!r}', ResourceWarning, source=self)
            self._pipe.close()

    def abort(self):
        self._close(None)

    def _fatal_error(self, exc, message='Fatal error on pipe transport'):
        if isinstance(exc, OSError):
            if self._loop.get_debug():
                logger.debug('%r: %s', self, message, exc_info=True)
        else:
            self._loop.call_exception_handler({'message': message, 'exception': exc, 'transport': self, 'protocol': self._protocol})
        self._close(exc)

    def _close(self, exc=None):
        self._closing = True
        if self._buffer:
            self._loop._remove_writer(self._fileno)
        self._buffer.clear()
        self._loop._remove_reader(self._fileno)
        self._loop.call_soon(self._call_connection_lost, exc)

    def _call_connection_lost(self, exc):
        try:
            self._protocol.connection_lost(exc)
        finally:
            self._pipe.close()
            self._pipe = None
            self._protocol = None
            self._loop = None