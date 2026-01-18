import errno
import os
import socket
import struct
import sys
import traceback
import warnings
from . import _auth
from .charset import charset_by_name, charset_by_id
from .constants import CLIENT, COMMAND, CR, ER, FIELD_TYPE, SERVER_STATUS
from . import converters
from .cursors import Cursor
from .optionfile import Parser
from .protocol import (
from . import err, VERSION_STRING
def _quote_bytes(self, s):
    if self.server_status & SERVER_STATUS.SERVER_STATUS_NO_BACKSLASH_ESCAPES:
        return "'{}'".format(s.replace(b"'", b"''").decode('ascii', 'surrogateescape'))
    return converters.escape_bytes(s)