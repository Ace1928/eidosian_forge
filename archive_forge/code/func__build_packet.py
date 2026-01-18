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
def _build_packet(self, payload):
    bsize = self.__block_size_out
    addlen = 4 if self.__etm_out else 8
    padding = 3 + bsize - (len(payload) + addlen) % bsize
    packet = struct.pack('>IB', len(payload) + padding + 1, padding)
    packet += payload
    if self.__sdctr_out or self.__block_engine_out is None:
        packet += zero_byte * padding
    else:
        packet += os.urandom(padding)
    return packet