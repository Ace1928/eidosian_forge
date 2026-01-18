import io
import os
import sys
import socket
import struct
import time
import tempfile
import itertools
from . import util
from . import AuthenticationError, BufferTooShort
from .context import reduction
class PipeListener(object):
    """
        Representation of a named pipe
        """

    def __init__(self, address, backlog=None):
        self._address = address
        self._handle_queue = [self._new_handle(first=True)]
        self._last_accepted = None
        util.sub_debug('listener created with address=%r', self._address)
        self.close = util.Finalize(self, PipeListener._finalize_pipe_listener, args=(self._handle_queue, self._address), exitpriority=0)

    def _new_handle(self, first=False):
        flags = _winapi.PIPE_ACCESS_DUPLEX | _winapi.FILE_FLAG_OVERLAPPED
        if first:
            flags |= _winapi.FILE_FLAG_FIRST_PIPE_INSTANCE
        return _winapi.CreateNamedPipe(self._address, flags, _winapi.PIPE_TYPE_MESSAGE | _winapi.PIPE_READMODE_MESSAGE | _winapi.PIPE_WAIT, _winapi.PIPE_UNLIMITED_INSTANCES, BUFSIZE, BUFSIZE, _winapi.NMPWAIT_WAIT_FOREVER, _winapi.NULL)

    def accept(self):
        self._handle_queue.append(self._new_handle())
        handle = self._handle_queue.pop(0)
        try:
            ov = _winapi.ConnectNamedPipe(handle, overlapped=True)
        except OSError as e:
            if e.winerror != _winapi.ERROR_NO_DATA:
                raise
        else:
            try:
                res = _winapi.WaitForMultipleObjects([ov.event], False, INFINITE)
            except:
                ov.cancel()
                _winapi.CloseHandle(handle)
                raise
            finally:
                _, err = ov.GetOverlappedResult(True)
                assert err == 0
        return PipeConnection(handle)

    @staticmethod
    def _finalize_pipe_listener(queue, address):
        util.sub_debug('closing listener with address=%r', address)
        for handle in queue:
            _winapi.CloseHandle(handle)