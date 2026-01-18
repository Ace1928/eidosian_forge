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
def _write_object(self, value):
    ref = self._getrefnum(value)
    self._object_offsets[ref] = self._fp.tell()
    if value is None:
        self._fp.write(b'\x00')
    elif value is False:
        self._fp.write(b'\x08')
    elif value is True:
        self._fp.write(b'\t')
    elif isinstance(value, int):
        if value < 0:
            try:
                self._fp.write(struct.pack('>Bq', 19, value))
            except struct.error:
                raise OverflowError(value) from None
        elif value < 1 << 8:
            self._fp.write(struct.pack('>BB', 16, value))
        elif value < 1 << 16:
            self._fp.write(struct.pack('>BH', 17, value))
        elif value < 1 << 32:
            self._fp.write(struct.pack('>BL', 18, value))
        elif value < 1 << 63:
            self._fp.write(struct.pack('>BQ', 19, value))
        elif value < 1 << 64:
            self._fp.write(b'\x14' + value.to_bytes(16, 'big', signed=True))
        else:
            raise OverflowError(value)
    elif isinstance(value, float):
        self._fp.write(struct.pack('>Bd', 35, value))
    elif isinstance(value, datetime.datetime):
        f = (value - datetime.datetime(2001, 1, 1)).total_seconds()
        self._fp.write(struct.pack('>Bd', 51, f))
    elif isinstance(value, (bytes, bytearray)):
        self._write_size(64, len(value))
        self._fp.write(value)
    elif isinstance(value, str):
        try:
            t = value.encode('ascii')
            self._write_size(80, len(value))
        except UnicodeEncodeError:
            t = value.encode('utf-16be')
            self._write_size(96, len(t) // 2)
        self._fp.write(t)
    elif isinstance(value, UID):
        if value.data < 0:
            raise ValueError('UIDs must be positive')
        elif value.data < 1 << 8:
            self._fp.write(struct.pack('>BB', 128, value))
        elif value.data < 1 << 16:
            self._fp.write(struct.pack('>BH', 129, value))
        elif value.data < 1 << 32:
            self._fp.write(struct.pack('>BL', 131, value))
        elif value.data < 1 << 64:
            self._fp.write(struct.pack('>BQ', 135, value))
        else:
            raise OverflowError(value)
    elif isinstance(value, (list, tuple)):
        refs = [self._getrefnum(o) for o in value]
        s = len(refs)
        self._write_size(160, s)
        self._fp.write(struct.pack('>' + self._ref_format * s, *refs))
    elif isinstance(value, dict):
        keyRefs, valRefs = ([], [])
        if self._sort_keys:
            rootItems = sorted(value.items())
        else:
            rootItems = value.items()
        for k, v in rootItems:
            if not isinstance(k, str):
                if self._skipkeys:
                    continue
                raise TypeError('keys must be strings')
            keyRefs.append(self._getrefnum(k))
            valRefs.append(self._getrefnum(v))
        s = len(keyRefs)
        self._write_size(208, s)
        self._fp.write(struct.pack('>' + self._ref_format * s, *keyRefs))
        self._fp.write(struct.pack('>' + self._ref_format * s, *valRefs))
    else:
        raise TypeError(value)