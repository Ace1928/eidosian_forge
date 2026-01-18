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
def _read_load_local_packet(self, first_packet):
    if not self.connection._local_infile:
        raise RuntimeError('**WARN**: Received LOAD_LOCAL packet but local_infile option is false.')
    load_packet = LoadLocalPacketWrapper(first_packet)
    sender = LoadLocalFile(load_packet.filename, self.connection)
    try:
        sender.send_data()
    except:
        self.connection._read_packet()
        raise
    ok_packet = self.connection._read_packet()
    if not ok_packet.is_ok_packet():
        raise err.OperationalError(CR.CR_COMMANDS_OUT_OF_SYNC, 'Commands Out of Sync')
    self._read_ok_packet(ok_packet)