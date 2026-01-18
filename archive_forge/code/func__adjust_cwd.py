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
def _adjust_cwd(self, path):
    """
        Return an adjusted path if we're emulating a "current working
        directory" for the server.
        """
    path = b(path)
    if self._cwd is None:
        return path
    if len(path) and path[0:1] == b_slash:
        return path
    if self._cwd == b_slash:
        return self._cwd + path
    return self._cwd + b_slash + path