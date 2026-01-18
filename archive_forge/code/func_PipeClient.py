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
def PipeClient(address):
    """
        Return a connection object connected to the pipe given by `address`
        """
    t = _init_timeout()
    while 1:
        try:
            _winapi.WaitNamedPipe(address, 1000)
            h = _winapi.CreateFile(address, _winapi.GENERIC_READ | _winapi.GENERIC_WRITE, 0, _winapi.NULL, _winapi.OPEN_EXISTING, _winapi.FILE_FLAG_OVERLAPPED, _winapi.NULL)
        except OSError as e:
            if e.winerror not in (_winapi.ERROR_SEM_TIMEOUT, _winapi.ERROR_PIPE_BUSY) or _check_timeout(t):
                raise
        else:
            break
    else:
        raise
    _winapi.SetNamedPipeHandleState(h, _winapi.PIPE_READMODE_MESSAGE, None, None)
    return PipeConnection(h)