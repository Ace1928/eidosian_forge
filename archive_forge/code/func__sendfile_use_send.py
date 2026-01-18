fromfd() -- create a socket object from an open file descriptor [*]
fromshare() -- create a socket object from data received from socket.share() [*]
import _socket
from _socket import *
import os, sys, io, selectors
from enum import IntEnum, IntFlag
def _sendfile_use_send(self, file, offset=0, count=None):
    self._check_sendfile_params(file, offset, count)
    if self.gettimeout() == 0:
        raise ValueError('non-blocking sockets are not supported')
    if offset:
        file.seek(offset)
    blocksize = min(count, 8192) if count else 8192
    total_sent = 0
    file_read = file.read
    sock_send = self.send
    try:
        while True:
            if count:
                blocksize = min(count - total_sent, blocksize)
                if blocksize <= 0:
                    break
            data = memoryview(file_read(blocksize))
            if not data:
                break
            while True:
                try:
                    sent = sock_send(data)
                except BlockingIOError:
                    continue
                else:
                    total_sent += sent
                    if sent < len(data):
                        data = data[sent:]
                    else:
                        break
        return total_sent
    finally:
        if total_sent > 0 and hasattr(file, 'seek'):
            file.seek(offset + total_sent)