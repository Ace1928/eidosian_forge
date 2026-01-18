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
class SFTP(SFTPClient):
    """
    An alias for `.SFTPClient` for backwards compatibility.
    """
    pass