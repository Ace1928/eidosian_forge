import errno
import os
import socket
import sys
import ovs.poller
import ovs.socket_util
import ovs.vlog
def __wait_windows(self, poller, wait):
    if self.socket is not None:
        if wait == Stream.W_RECV:
            mask = win32file.FD_READ | win32file.FD_ACCEPT | win32file.FD_CLOSE
            event = ovs.poller.POLLIN
        else:
            mask = win32file.FD_WRITE | win32file.FD_CONNECT | win32file.FD_CLOSE
            event = ovs.poller.POLLOUT
        try:
            win32file.WSAEventSelect(self.socket, self._wevent, mask)
        except pywintypes.error as e:
            vlog.err('failed to associate events with socket: %s' % e.strerror)
        poller.fd_wait(self._wevent, event)
    elif wait == Stream.W_RECV:
        if self._read:
            poller.fd_wait(self._read.hEvent, ovs.poller.POLLIN)
    elif wait == Stream.W_SEND:
        if self._write:
            poller.fd_wait(self._write.hEvent, ovs.poller.POLLOUT)
    elif wait == Stream.W_CONNECT:
        return