import errno
import getpass
import logging
import os
import socket
import subprocess
import sys
from binascii import hexlify
from typing import Dict, Optional, Set, Tuple, Type
from .. import bedding, config, errors, osutils, trace, ui
import weakref
class SSHConnection:
    """Abstract base class for SSH connections."""

    def get_sock_or_pipes(self):
        """Returns a (kind, io_object) pair.

        If kind == 'socket', then io_object is a socket.

        If kind == 'pipes', then io_object is a pair of file-like objects
        (read_from, write_to).
        """
        raise NotImplementedError(self.get_sock_or_pipes)

    def close(self):
        raise NotImplementedError(self.close)