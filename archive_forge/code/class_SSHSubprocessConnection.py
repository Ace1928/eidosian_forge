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
class SSHSubprocessConnection(SSHConnection):
    """A connection to an ssh subprocess via pipes or a socket.

    This class is also socket-like enough to be used with
    SocketAsChannelAdapter (it has 'send' and 'recv' methods).
    """

    def __init__(self, proc, sock=None):
        """Constructor.

        :param proc: a subprocess.Popen
        :param sock: if proc.stdin/out is a socket from a socketpair, then sock
            should breezy's half of that socketpair.  If not passed, proc's
            stdin/out is assumed to be ordinary pipes.
        """
        self.proc = proc
        self._sock = sock

        def terminate(ref):
            _subproc_weakrefs.remove(ref)
            _close_ssh_proc(proc, sock)
        _subproc_weakrefs.add(weakref.ref(self, terminate))

    def send(self, data):
        if self._sock is not None:
            return self._sock.send(data)
        else:
            return os.write(self.proc.stdin.fileno(), data)

    def recv(self, count):
        if self._sock is not None:
            return self._sock.recv(count)
        else:
            return os.read(self.proc.stdout.fileno(), count)

    def close(self):
        _close_ssh_proc(self.proc, self._sock)

    def get_sock_or_pipes(self):
        if self._sock is not None:
            return ('socket', self._sock)
        else:
            return ('pipes', (self.proc.stdout, self.proc.stdin))