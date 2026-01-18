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
def _request_authentication(self):
    if int(self.server_version.split('.', 1)[0]) >= 5:
        self.client_flag |= CLIENT.MULTI_RESULTS
    if self.user is None:
        raise ValueError('Did not specify a username')
    charset_id = charset_by_name(self.charset).id
    if isinstance(self.user, str):
        self.user = self.user.encode(self.encoding)
    data_init = struct.pack('<iIB23s', self.client_flag, MAX_PACKET_LEN, charset_id, b'')
    if self.ssl and self.server_capabilities & CLIENT.SSL:
        self.write_packet(data_init)
        self._sock = self.ctx.wrap_socket(self._sock, server_hostname=self.host)
        self._rfile = self._sock.makefile('rb')
        self._secure = True
    data = data_init + self.user + b'\x00'
    authresp = b''
    plugin_name = None
    if self._auth_plugin_name == '':
        plugin_name = b''
        authresp = _auth.scramble_native_password(self.password, self.salt)
    elif self._auth_plugin_name == 'mysql_native_password':
        plugin_name = b'mysql_native_password'
        authresp = _auth.scramble_native_password(self.password, self.salt)
    elif self._auth_plugin_name == 'caching_sha2_password':
        plugin_name = b'caching_sha2_password'
        if self.password:
            if DEBUG:
                print('caching_sha2: trying fast path')
            authresp = _auth.scramble_caching_sha2(self.password, self.salt)
        elif DEBUG:
            print('caching_sha2: empty password')
    elif self._auth_plugin_name == 'sha256_password':
        plugin_name = b'sha256_password'
        if self.ssl and self.server_capabilities & CLIENT.SSL:
            authresp = self.password + b'\x00'
        elif self.password:
            authresp = b'\x01'
        else:
            authresp = b'\x00'
    if self.server_capabilities & CLIENT.PLUGIN_AUTH_LENENC_CLIENT_DATA:
        data += _lenenc_int(len(authresp)) + authresp
    elif self.server_capabilities & CLIENT.SECURE_CONNECTION:
        data += struct.pack('B', len(authresp)) + authresp
    else:
        data += authresp + b'\x00'
    if self.db and self.server_capabilities & CLIENT.CONNECT_WITH_DB:
        if isinstance(self.db, str):
            self.db = self.db.encode(self.encoding)
        data += self.db + b'\x00'
    if self.server_capabilities & CLIENT.PLUGIN_AUTH:
        data += (plugin_name or b'') + b'\x00'
    if self.server_capabilities & CLIENT.CONNECT_ATTRS:
        connect_attrs = b''
        for k, v in self._connect_attrs.items():
            k = k.encode('utf-8')
            connect_attrs += _lenenc_int(len(k)) + k
            v = v.encode('utf-8')
            connect_attrs += _lenenc_int(len(v)) + v
        data += _lenenc_int(len(connect_attrs)) + connect_attrs
    self.write_packet(data)
    auth_packet = self._read_packet()
    if auth_packet.is_auth_switch_request():
        if DEBUG:
            print('received auth switch')
        auth_packet.read_uint8()
        plugin_name = auth_packet.read_string()
        if self.server_capabilities & CLIENT.PLUGIN_AUTH and plugin_name is not None:
            auth_packet = self._process_auth(plugin_name, auth_packet)
        else:
            raise err.OperationalError('received unknown auth switch request')
    elif auth_packet.is_extra_auth_data():
        if DEBUG:
            print('received extra data')
        if self._auth_plugin_name == 'caching_sha2_password':
            auth_packet = _auth.caching_sha2_password_auth(self, auth_packet)
        elif self._auth_plugin_name == 'sha256_password':
            auth_packet = _auth.sha256_password_auth(self, auth_packet)
        else:
            raise err.OperationalError('Received extra packet for auth method %r', self._auth_plugin_name)
    if DEBUG:
        print('Succeed to auth')