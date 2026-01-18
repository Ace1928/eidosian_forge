from binascii import hexlify
from collections import deque
import socket
import threading
import time
from paramiko.common import DEBUG, io_sleep
from paramiko.file import BufferedFile
from paramiko.util import u
from paramiko.sftp import (
from paramiko.sftp_attr import SFTPAttributes
def _data_in_prefetch_requests(self, offset, size):
    k = [x for x in list(self._prefetch_extents.values()) if x[0] <= offset]
    if len(k) == 0:
        return False
    k.sort(key=lambda x: x[0])
    buf_offset, buf_size = k[-1]
    if buf_offset + buf_size <= offset:
        return False
    if buf_offset + buf_size >= offset + size:
        return True
    return self._data_in_prefetch_requests(buf_offset + buf_size, offset + size - buf_offset - buf_size)