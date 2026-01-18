import os
import socket
import _socket
from multiprocessing.connection import Connection
from multiprocessing.context import get_spawning_popen
from .reduction import register
def _mk_inheritable(fd):
    os.set_inheritable(fd, True)
    return fd