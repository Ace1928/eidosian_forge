import email.parser
import email.message
import errno
import http
import io
import re
import socket
import sys
import collections.abc
from urllib.parse import urlsplit
def _output(self, s):
    """Add a line of output to the current request buffer.

        Assumes that the line does *not* end with \\r\\n.
        """
    self._buffer.append(s)