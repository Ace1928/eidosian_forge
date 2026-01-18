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
def _get_descriptions(self):
    """Read a column descriptor packet for each column in the result."""
    self.fields = []
    self.converters = []
    use_unicode = self.connection.use_unicode
    conn_encoding = self.connection.encoding
    description = []
    for i in range(self.field_count):
        field = self.connection._read_packet(FieldDescriptorPacket)
        self.fields.append(field)
        description.append(field.description())
        field_type = field.type_code
        if use_unicode:
            if field_type == FIELD_TYPE.JSON:
                encoding = conn_encoding
            elif field_type in TEXT_TYPES:
                if field.charsetnr == 63:
                    encoding = None
                else:
                    encoding = conn_encoding
            else:
                encoding = 'ascii'
        else:
            encoding = None
        converter = self.connection.decoders.get(field_type)
        if converter is converters.through:
            converter = None
        if DEBUG:
            print(f'DEBUG: field={field}, converter={converter}')
        self.converters.append((encoding, converter))
    eof_packet = self.connection._read_packet()
    assert eof_packet.is_eof_packet(), 'Protocol error, expecting EOF'
    self.description = tuple(description)