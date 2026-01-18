import binascii
import codecs
import datetime
import enum
from io import BytesIO
import itertools
import os
import re
import struct
from xml.parsers.expat import ParserCreate
class _BinaryPlistParser:
    """
    Read or write a binary plist file, following the description of the binary
    format.  Raise InvalidFileException in case of error, otherwise return the
    root object.

    see also: http://opensource.apple.com/source/CF/CF-744.18/CFBinaryPList.c
    """

    def __init__(self, dict_type):
        self._dict_type = dict_type

    def parse(self, fp):
        try:
            self._fp = fp
            self._fp.seek(-32, os.SEEK_END)
            trailer = self._fp.read(32)
            if len(trailer) != 32:
                raise InvalidFileException()
            offset_size, self._ref_size, num_objects, top_object, offset_table_offset = struct.unpack('>6xBBQQQ', trailer)
            self._fp.seek(offset_table_offset)
            self._object_offsets = self._read_ints(num_objects, offset_size)
            self._objects = [_undefined] * num_objects
            return self._read_object(top_object)
        except (OSError, IndexError, struct.error, OverflowError, ValueError):
            raise InvalidFileException()

    def _get_size(self, tokenL):
        """ return the size of the next object."""
        if tokenL == 15:
            m = self._fp.read(1)[0] & 3
            s = 1 << m
            f = '>' + _BINARY_FORMAT[s]
            return struct.unpack(f, self._fp.read(s))[0]
        return tokenL

    def _read_ints(self, n, size):
        data = self._fp.read(size * n)
        if size in _BINARY_FORMAT:
            return struct.unpack(f'>{n}{_BINARY_FORMAT[size]}', data)
        else:
            if not size or len(data) != size * n:
                raise InvalidFileException()
            return tuple((int.from_bytes(data[i:i + size], 'big') for i in range(0, size * n, size)))

    def _read_refs(self, n):
        return self._read_ints(n, self._ref_size)

    def _read_object(self, ref):
        """
        read the object by reference.

        May recursively read sub-objects (content of an array/dict/set)
        """
        result = self._objects[ref]
        if result is not _undefined:
            return result
        offset = self._object_offsets[ref]
        self._fp.seek(offset)
        token = self._fp.read(1)[0]
        tokenH, tokenL = (token & 240, token & 15)
        if token == 0:
            result = None
        elif token == 8:
            result = False
        elif token == 9:
            result = True
        elif token == 15:
            result = b''
        elif tokenH == 16:
            result = int.from_bytes(self._fp.read(1 << tokenL), 'big', signed=tokenL >= 3)
        elif token == 34:
            result = struct.unpack('>f', self._fp.read(4))[0]
        elif token == 35:
            result = struct.unpack('>d', self._fp.read(8))[0]
        elif token == 51:
            f = struct.unpack('>d', self._fp.read(8))[0]
            result = datetime.datetime(2001, 1, 1) + datetime.timedelta(seconds=f)
        elif tokenH == 64:
            s = self._get_size(tokenL)
            result = self._fp.read(s)
            if len(result) != s:
                raise InvalidFileException()
        elif tokenH == 80:
            s = self._get_size(tokenL)
            data = self._fp.read(s)
            if len(data) != s:
                raise InvalidFileException()
            result = data.decode('ascii')
        elif tokenH == 96:
            s = self._get_size(tokenL) * 2
            data = self._fp.read(s)
            if len(data) != s:
                raise InvalidFileException()
            result = data.decode('utf-16be')
        elif tokenH == 128:
            result = UID(int.from_bytes(self._fp.read(1 + tokenL), 'big'))
        elif tokenH == 160:
            s = self._get_size(tokenL)
            obj_refs = self._read_refs(s)
            result = []
            self._objects[ref] = result
            result.extend((self._read_object(x) for x in obj_refs))
        elif tokenH == 208:
            s = self._get_size(tokenL)
            key_refs = self._read_refs(s)
            obj_refs = self._read_refs(s)
            result = self._dict_type()
            self._objects[ref] = result
            try:
                for k, o in zip(key_refs, obj_refs):
                    result[self._read_object(k)] = self._read_object(o)
            except TypeError:
                raise InvalidFileException()
        else:
            raise InvalidFileException()
        self._objects[ref] = result
        return result