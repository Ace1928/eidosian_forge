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
def _get_server_information(self):
    i = 0
    packet = self._read_packet()
    data = packet.get_all_data()
    self.protocol_version = data[i]
    i += 1
    server_end = data.find(b'\x00', i)
    self.server_version = data[i:server_end].decode('latin1')
    i = server_end + 1
    self.server_thread_id = struct.unpack('<I', data[i:i + 4])
    i += 4
    self.salt = data[i:i + 8]
    i += 9
    self.server_capabilities = struct.unpack('<H', data[i:i + 2])[0]
    i += 2
    if len(data) >= i + 6:
        lang, stat, cap_h, salt_len = struct.unpack('<BHHB', data[i:i + 6])
        i += 6
        self.server_language = lang
        try:
            self.server_charset = charset_by_id(lang).name
        except KeyError:
            self.server_charset = None
        self.server_status = stat
        if DEBUG:
            print('server_status: %x' % stat)
        self.server_capabilities |= cap_h << 16
        if DEBUG:
            print('salt_len:', salt_len)
        salt_len = max(12, salt_len - 9)
    i += 10
    if len(data) >= i + salt_len:
        self.salt += data[i:i + salt_len]
        i += salt_len
    i += 1
    if self.server_capabilities & CLIENT.PLUGIN_AUTH and len(data) >= i:
        server_end = data.find(b'\x00', i)
        if server_end < 0:
            self._auth_plugin_name = data[i:].decode('utf-8')
        else:
            self._auth_plugin_name = data[i:server_end].decode('utf-8')