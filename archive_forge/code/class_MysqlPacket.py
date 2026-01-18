from .charset import MBLENGTH
from .constants import FIELD_TYPE, SERVER_STATUS
from . import err
import struct
import sys
class MysqlPacket:
    """Representation of a MySQL response packet.

    Provides an interface for reading/parsing the packet results.
    """
    __slots__ = ('_position', '_data')

    def __init__(self, data, encoding):
        self._position = 0
        self._data = data

    def get_all_data(self):
        return self._data

    def read(self, size):
        """Read the first 'size' bytes in packet and advance cursor past them."""
        result = self._data[self._position:self._position + size]
        if len(result) != size:
            error = 'Result length not requested length:\nExpected=%s.  Actual=%s.  Position: %s.  Data Length: %s' % (size, len(result), self._position, len(self._data))
            if DEBUG:
                print(error)
                self.dump()
            raise AssertionError(error)
        self._position += size
        return result

    def read_all(self):
        """Read all remaining data in the packet.

        (Subsequent read() will return errors.)
        """
        result = self._data[self._position:]
        self._position = None
        return result

    def advance(self, length):
        """Advance the cursor in data buffer 'length' bytes."""
        new_position = self._position + length
        if new_position < 0 or new_position > len(self._data):
            raise Exception('Invalid advance amount (%s) for cursor.  Position=%s' % (length, new_position))
        self._position = new_position

    def rewind(self, position=0):
        """Set the position of the data buffer cursor to 'position'."""
        if position < 0 or position > len(self._data):
            raise Exception('Invalid position to rewind cursor to: %s.' % position)
        self._position = position

    def get_bytes(self, position, length=1):
        """Get 'length' bytes starting at 'position'.

        Position is start of payload (first four packet header bytes are not
        included) starting at index '0'.

        No error checking is done.  If requesting outside end of buffer
        an empty string (or string shorter than 'length') may be returned!
        """
        return self._data[position:position + length]

    def read_uint8(self):
        result = self._data[self._position]
        self._position += 1
        return result

    def read_uint16(self):
        result = struct.unpack_from('<H', self._data, self._position)[0]
        self._position += 2
        return result

    def read_uint24(self):
        low, high = struct.unpack_from('<HB', self._data, self._position)
        self._position += 3
        return low + (high << 16)

    def read_uint32(self):
        result = struct.unpack_from('<I', self._data, self._position)[0]
        self._position += 4
        return result

    def read_uint64(self):
        result = struct.unpack_from('<Q', self._data, self._position)[0]
        self._position += 8
        return result

    def read_string(self):
        end_pos = self._data.find(b'\x00', self._position)
        if end_pos < 0:
            return None
        result = self._data[self._position:end_pos]
        self._position = end_pos + 1
        return result

    def read_length_encoded_integer(self):
        """Read a 'Length Coded Binary' number from the data buffer.

        Length coded numbers can be anywhere from 1 to 9 bytes depending
        on the value of the first byte.
        """
        c = self.read_uint8()
        if c == NULL_COLUMN:
            return None
        if c < UNSIGNED_CHAR_COLUMN:
            return c
        elif c == UNSIGNED_SHORT_COLUMN:
            return self.read_uint16()
        elif c == UNSIGNED_INT24_COLUMN:
            return self.read_uint24()
        elif c == UNSIGNED_INT64_COLUMN:
            return self.read_uint64()

    def read_length_coded_string(self):
        """Read a 'Length Coded String' from the data buffer.

        A 'Length Coded String' consists first of a length coded
        (unsigned, positive) integer represented in 1-9 bytes followed by
        that many bytes of binary data.  (For example "cat" would be "3cat".)
        """
        length = self.read_length_encoded_integer()
        if length is None:
            return None
        return self.read(length)

    def read_struct(self, fmt):
        s = struct.Struct(fmt)
        result = s.unpack_from(self._data, self._position)
        self._position += s.size
        return result

    def is_ok_packet(self):
        return self._data[0] == 0 and len(self._data) >= 7

    def is_eof_packet(self):
        return self._data[0] == 254 and len(self._data) < 9

    def is_auth_switch_request(self):
        return self._data[0] == 254

    def is_extra_auth_data(self):
        return self._data[0] == 1

    def is_resultset_packet(self):
        field_count = self._data[0]
        return 1 <= field_count <= 250

    def is_load_local_packet(self):
        return self._data[0] == 251

    def is_error_packet(self):
        return self._data[0] == 255

    def check_error(self):
        if self.is_error_packet():
            self.raise_for_error()

    def raise_for_error(self):
        self.rewind()
        self.advance(1)
        errno = self.read_uint16()
        if DEBUG:
            print('errno =', errno)
        err.raise_mysql_exception(self._data)

    def dump(self):
        dump_packet(self._data)