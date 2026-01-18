import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
class IMAP4_stream(IMAP4):
    """IMAP4 client class over a stream

    Instantiate with: IMAP4_stream(command)

            "command" - a string that can be passed to subprocess.Popen()

    for more documentation see the docstring of the parent class IMAP4.
    """

    def __init__(self, command):
        self.command = command
        IMAP4.__init__(self)

    def open(self, host=None, port=None, timeout=None):
        """Setup a stream connection.
        This connection will be used by the routines:
            read, readline, send, shutdown.
        """
        self.host = None
        self.port = None
        self.sock = None
        self.file = None
        self.process = subprocess.Popen(self.command, bufsize=DEFAULT_BUFFER_SIZE, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, close_fds=True)
        self.writefile = self.process.stdin
        self.readfile = self.process.stdout

    def read(self, size):
        """Read 'size' bytes from remote."""
        return self.readfile.read(size)

    def readline(self):
        """Read line from remote."""
        return self.readfile.readline()

    def send(self, data):
        """Send data to remote."""
        self.writefile.write(data)
        self.writefile.flush()

    def shutdown(self):
        """Close I/O established in "open"."""
        self.readfile.close()
        self.writefile.close()
        self.process.wait()