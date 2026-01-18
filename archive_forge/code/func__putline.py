import errno
import re
import socket
import sys
def _putline(self, line):
    if self._debugging > 1:
        print('*put*', repr(line))
    sys.audit('poplib.putline', self, line)
    self.sock.sendall(line + CRLF)