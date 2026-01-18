import io, logging, socket, os, pickle, struct, time, re
from stat import ST_DEV, ST_INO, ST_MTIME
import queue
import threading
import copy
def _connect_unixsocket(self, address):
    use_socktype = self.socktype
    if use_socktype is None:
        use_socktype = socket.SOCK_DGRAM
    self.socket = socket.socket(socket.AF_UNIX, use_socktype)
    try:
        self.socket.connect(address)
        self.socktype = use_socktype
    except OSError:
        self.socket.close()
        if self.socktype is not None:
            raise
        use_socktype = socket.SOCK_STREAM
        self.socket = socket.socket(socket.AF_UNIX, use_socktype)
        try:
            self.socket.connect(address)
            self.socktype = use_socktype
        except OSError:
            self.socket.close()
            raise