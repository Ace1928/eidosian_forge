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
class ChannelStderrFile(ChannelFile):
    """
    A file-like wrapper around `.Channel` stderr.

    See `Channel.makefile_stderr` for details.
    """

    def _read(self, size):
        return self.channel.recv_stderr(size)

    def _write(self, data):
        self.channel.sendall_stderr(data)
        return len(data)