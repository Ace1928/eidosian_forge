import errno
import os
import socket
import struct
import threading
import time
from hmac import HMAC
from paramiko import util
from paramiko.common import (
from paramiko.util import u
from paramiko.ssh_exception import SSHException, ProxyCommandFailure
from paramiko.message import Message
def _read_timeout(self, timeout):
    start = time.time()
    while True:
        try:
            x = self.__socket.recv(128)
            if len(x) == 0:
                raise EOFError()
            break
        except socket.timeout:
            pass
        if self.__closed:
            raise EOFError()
        now = time.time()
        if now - start >= timeout:
            raise socket.timeout()
    return x