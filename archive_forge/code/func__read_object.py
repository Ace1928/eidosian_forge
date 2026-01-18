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