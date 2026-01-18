import collections
import subprocess
import warnings
from . import protocols
from . import transports
from .log import logger
class WriteSubprocessPipeProto(protocols.BaseProtocol):

    def __init__(self, proc, fd):
        self.proc = proc
        self.fd = fd
        self.pipe = None
        self.disconnected = False

    def connection_made(self, transport):
        self.pipe = transport

    def __repr__(self):
        return f'<{self.__class__.__name__} fd={self.fd} pipe={self.pipe!r}>'

    def connection_lost(self, exc):
        self.disconnected = True
        self.proc._pipe_connection_lost(self.fd, exc)
        self.proc = None

    def pause_writing(self):
        self.proc._protocol.pause_writing()

    def resume_writing(self):
        self.proc._protocol.resume_writing()