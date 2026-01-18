import os
import shlex
import signal
from select import select
import socket
import time
from paramiko.ssh_exception import ProxyCommandFailure
from paramiko.util import ClosingContextManager
class ProxyCommand(ClosingContextManager):
    """
    Wraps a subprocess running ProxyCommand-driven programs.

    This class implements a the socket-like interface needed by the
    `.Transport` and `.Packetizer` classes. Using this class instead of a
    regular socket makes it possible to talk with a Popen'd command that will
    proxy traffic between the client and a server hosted in another machine.

    Instances of this class may be used as context managers.
    """

    def __init__(self, command_line):
        """
        Create a new CommandProxy instance. The instance created by this
        class can be passed as an argument to the `.Transport` class.

        :param str command_line:
            the command that should be executed and used as the proxy.
        """
        if subprocess is None:
            raise subprocess_import_error
        self.cmd = shlex.split(command_line)
        self.process = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.timeout = None

    def send(self, content):
        """
        Write the content received from the SSH client to the standard
        input of the forked command.

        :param str content: string to be sent to the forked command
        """
        try:
            self.process.stdin.write(content)
        except IOError as e:
            raise ProxyCommandFailure(' '.join(self.cmd), e.strerror)
        return len(content)

    def recv(self, size):
        """
        Read from the standard output of the forked program.

        :param int size: how many chars should be read

        :return: the string of bytes read, which may be shorter than requested
        """
        try:
            buffer = b''
            start = time.time()
            while len(buffer) < size:
                select_timeout = None
                if self.timeout is not None:
                    elapsed = time.time() - start
                    if elapsed >= self.timeout:
                        raise socket.timeout()
                    select_timeout = self.timeout - elapsed
                r, w, x = select([self.process.stdout], [], [], select_timeout)
                if r and r[0] == self.process.stdout:
                    buffer += os.read(self.process.stdout.fileno(), size - len(buffer))
            return buffer
        except socket.timeout:
            if buffer:
                return buffer
            raise
        except IOError as e:
            raise ProxyCommandFailure(' '.join(self.cmd), e.strerror)

    def close(self):
        os.kill(self.process.pid, signal.SIGTERM)

    @property
    def closed(self):
        return self.process.returncode is not None

    @property
    def _closed(self):
        return self.closed

    def settimeout(self, timeout):
        self.timeout = timeout