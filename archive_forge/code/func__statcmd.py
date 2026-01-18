import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
def _statcmd(self, line):
    """Internal: process a STAT, NEXT or LAST command."""
    resp = self._shortcmd(line)
    return self._statparse(resp)