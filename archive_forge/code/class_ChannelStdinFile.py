import binascii
import os
import socket
import time
import threading
from functools import wraps
from paramiko import util
from paramiko.common import (
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
from paramiko.file import BufferedFile
from paramiko.buffered_pipe import BufferedPipe, PipeTimeout
from paramiko import pipe
from paramiko.util import ClosingContextManager
class ChannelStdinFile(ChannelFile):
    """
    A file-like wrapper around `.Channel` stdin.

    See `Channel.makefile_stdin` for details.
    """

    def close(self):
        super().close()
        self.channel.shutdown_write()