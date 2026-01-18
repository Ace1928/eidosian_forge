import errno
import os
import socket
import sys
import ovs.poller
import ovs.socket_util
import ovs.vlog
def __accept_windows(self):
    if self.connect_pending:
        try:
            winutils.get_overlapped_result(self.pipe, self.connect, False)
        except pywintypes.error as e:
            if e.winerror == winutils.winerror.ERROR_IO_INCOMPLETE:
                self.connect_pending = True
                return (errno.EAGAIN, None)
            else:
                if self.pipe:
                    win32pipe.DisconnectNamedPipe(self.pipe)
                return (errno.EINVAL, None)
        self.connect_pending = False
    error = winutils.connect_named_pipe(self.pipe, self.connect)
    if error:
        if error == winutils.winerror.ERROR_IO_PENDING:
            self.connect_pending = True
            return (errno.EAGAIN, None)
        elif error != winutils.winerror.ERROR_PIPE_CONNECTED:
            if self.pipe:
                win32pipe.DisconnectNamedPipe(self.pipe)
            self.connect_pending = False
            return (errno.EINVAL, None)
        else:
            win32event.SetEvent(self.connect.hEvent)
    npipe = winutils.create_named_pipe(self._pipename)
    if not npipe:
        return (errno.ENOENT, None)
    old_pipe = self.pipe
    self.pipe = npipe
    winutils.win32event.ResetEvent(self.connect.hEvent)
    return (0, Stream(None, self.name, 0, pipe=old_pipe))