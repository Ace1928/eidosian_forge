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
class LoopbackVendor(SSHVendor):
    """SSH "vendor" that connects over a plain TCP socket, not SSH."""

    def connect_sftp(self, username, password, host, port):
        sock = socket.socket()
        try:
            sock.connect((host, port))
        except OSError as e:
            self._raise_connection_error(host, port=port, orig_error=e)
        return SFTPClient(SocketAsChannelAdapter(sock))