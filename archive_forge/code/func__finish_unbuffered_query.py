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
def _finish_unbuffered_query(self):
    while self.unbuffered_active:
        try:
            packet = self.connection._read_packet()
        except err.OperationalError as e:
            if e.args[0] in (ER.QUERY_TIMEOUT, ER.STATEMENT_TIMEOUT):
                self.unbuffered_active = False
                self.connection = None
                return
            raise
        if self._check_packet_is_eof(packet):
            self.unbuffered_active = False
            self.connection = None