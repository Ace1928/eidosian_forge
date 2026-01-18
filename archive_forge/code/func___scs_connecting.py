import errno
import os
import socket
import sys
import ovs.poller
import ovs.socket_util
import ovs.vlog
def __scs_connecting(self):
    if self.socket is not None:
        retval = self.check_connection_completion(self.socket)
        assert retval != errno.EINPROGRESS
    elif sys.platform == 'win32':
        if self.retry_connect:
            try:
                self.pipe = winutils.create_file(self._pipename)
                self._retry_connect = False
                retval = 0
            except pywintypes.error as e:
                if e.winerror == winutils.winerror.ERROR_PIPE_BUSY:
                    retval = errno.EAGAIN
                else:
                    self._retry_connect = False
                    retval = errno.ENOENT
        else:
            retval = 0
    if retval == 0:
        self.state = Stream.__S_CONNECTED
    elif retval != errno.EAGAIN:
        self.state = Stream.__S_DISCONNECTED
        self.error = retval