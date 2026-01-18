import collections
import subprocess
import warnings
from . import protocols
from . import transports
from .log import logger
class ReadSubprocessPipeProto(WriteSubprocessPipeProto, protocols.Protocol):

    def data_received(self, data):
        self.proc._pipe_data_received(self.fd, data)