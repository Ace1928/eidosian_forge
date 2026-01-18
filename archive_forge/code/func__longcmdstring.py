import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
def _longcmdstring(self, line, file=None):
    """Internal: send a command and get the response plus following text.
        Same as _longcmd() and _getlongresp(), except that the returned `lines`
        are unicode strings rather than bytes objects.
        """
    self._putcmd(line)
    resp, list = self._getlongresp(file)
    return (resp, [line.decode(self.encoding, self.errors) for line in list])