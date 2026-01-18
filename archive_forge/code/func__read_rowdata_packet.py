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
def _read_rowdata_packet(self):
    """Read a rowdata packet for each data row in the result set."""
    rows = []
    while True:
        packet = self.connection._read_packet()
        if self._check_packet_is_eof(packet):
            self.connection = None
            break
        rows.append(self._read_row_from_packet(packet))
    self.affected_rows = len(rows)
    self.rows = tuple(rows)