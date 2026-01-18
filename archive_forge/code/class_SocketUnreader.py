import io
import os
class SocketUnreader(Unreader):

    def __init__(self, sock, max_chunk=8192):
        super().__init__()
        self.sock = sock
        self.mxchunk = max_chunk

    def chunk(self):
        return self.sock.recv(self.mxchunk)