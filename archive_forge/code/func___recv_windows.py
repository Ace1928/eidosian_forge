import errno
import os
import socket
import sys
import ovs.poller
import ovs.socket_util
import ovs.vlog
def __recv_windows(self, n):
    if self._read_pending:
        try:
            nBytesRead = winutils.get_overlapped_result(self.pipe, self._read, False)
            self._read_pending = False
        except pywintypes.error as e:
            if e.winerror == winutils.winerror.ERROR_IO_INCOMPLETE:
                self._read_pending = True
                return (errno.EAGAIN, '')
            elif e.winerror in winutils.pipe_disconnected_errors:
                return (0, '')
            else:
                return (errno.EINVAL, '')
    else:
        errCode, self._read_buffer = winutils.read_file(self.pipe, n, self._read)
        if errCode:
            if errCode == winutils.winerror.ERROR_IO_PENDING:
                self._read_pending = True
                return (errno.EAGAIN, '')
            elif errCode in winutils.pipe_disconnected_errors:
                return (0, '')
            else:
                return (errCode, '')
        try:
            nBytesRead = winutils.get_overlapped_result(self.pipe, self._read, False)
            winutils.win32event.SetEvent(self._read.hEvent)
        except pywintypes.error as e:
            if e.winerror in winutils.pipe_disconnected_errors:
                return (0, '')
            else:
                return (e.winerror, '')
    recvBuffer = self._read_buffer[:nBytesRead]
    return (0, bytes(recvBuffer))