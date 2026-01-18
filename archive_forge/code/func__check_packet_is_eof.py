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
def _check_packet_is_eof(self, packet):
    if not packet.is_eof_packet():
        return False
    wp = EOFPacketWrapper(packet)
    self.warning_count = wp.warning_count
    self.has_next = wp.has_next
    return True