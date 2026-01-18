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
def _async_response(self, t, msg, num):
    if t == CMD_STATUS:
        try:
            self.sftp._convert_status(msg)
        except Exception as e:
            self._saved_exception = e
        return
    if t != CMD_DATA:
        raise SFTPError('Expected data')
    data = msg.get_string()
    while True:
        with self._prefetch_lock:
            if num in self._prefetch_extents:
                offset, length = self._prefetch_extents[num]
                self._prefetch_data[offset] = data
                del self._prefetch_extents[num]
                if len(self._prefetch_extents) == 0:
                    self._prefetch_done = True
                break