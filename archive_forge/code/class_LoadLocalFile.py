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
class LoadLocalFile:

    def __init__(self, filename, connection):
        self.filename = filename
        self.connection = connection

    def send_data(self):
        """Send data packets from the local file to the server"""
        if not self.connection._sock:
            raise err.InterfaceError(0, '')
        conn: Connection = self.connection
        try:
            with open(self.filename, 'rb') as open_file:
                packet_size = min(conn.max_allowed_packet, 16 * 1024)
                while True:
                    chunk = open_file.read(packet_size)
                    if not chunk:
                        break
                    conn.write_packet(chunk)
        except OSError:
            raise err.OperationalError(ER.FILE_NOT_FOUND, f"Can't find file '{self.filename}'")
        finally:
            if not conn._closed:
                conn.write_packet(b'')