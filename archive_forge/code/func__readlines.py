import re
import struct
import sys
import eventlet
from eventlet import patcher
from eventlet.green import _socket_nodns
from eventlet.green import os
from eventlet.green import time
from eventlet.green import select
from eventlet.green import ssl
def _readlines(self):
    """Read the contents of the hosts file

        Return list of lines, comment lines and empty lines are
        excluded.

        Note that this performs disk I/O so can be blocking.
        """
    try:
        with open(self.fname, 'rb') as fp:
            fdata = fp.read()
    except OSError:
        return []
    udata = fdata.decode(errors='ignore')
    return filter(None, self.LINES_RE.findall(udata))