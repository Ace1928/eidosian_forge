from binascii import hexlify
import errno
import os
import stat
import threading
import time
import weakref
from paramiko import util
from paramiko.channel import Channel
from paramiko.message import Message
from paramiko.common import INFO, DEBUG, o777
from paramiko.sftp import (
from paramiko.sftp_attr import SFTPAttributes
from paramiko.ssh_exception import SSHException
from paramiko.sftp_file import SFTPFile
from paramiko.util import ClosingContextManager, b, u
def _transfer_with_callback(self, reader, writer, file_size, callback):
    size = 0
    while True:
        data = reader.read(32768)
        writer.write(data)
        size += len(data)
        if len(data) == 0:
            break
        if callback is not None:
            callback(size, file_size)
    return size