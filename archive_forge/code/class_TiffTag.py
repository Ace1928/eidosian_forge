from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
class TiffTag(object):
    """TIFF tag structure.

    Attributes
    ----------
    name : string
        Name of tag.
    code : int
        Decimal code of tag.
    dtype : str
        Datatype of tag data. One of TIFF DATA_FORMATS.
    count : int
        Number of values.
    value : various types
        Tag data as Python object.
    ImageSourceData : int
        Location of value in file.

    All attributes are read-only.

    """
    __slots__ = ('code', 'count', 'dtype', 'value', 'valueoffset')

    class Error(Exception):
        pass

    def __init__(self, parent, tagheader, **kwargs):
        """Initialize instance from tag header."""
        fh = parent.filehandle
        byteorder = parent.byteorder
        unpack = struct.unpack
        offsetsize = parent.offsetsize
        self.valueoffset = fh.tell() + offsetsize + 4
        code, type_ = unpack(parent.tagformat1, tagheader[:4])
        count, value = unpack(parent.tagformat2, tagheader[4:])
        try:
            dtype = TIFF.DATA_FORMATS[type_]
        except KeyError:
            raise TiffTag.Error('unknown tag data type %i' % type_)
        fmt = '%s%i%s' % (byteorder, count * int(dtype[0]), dtype[1])
        size = struct.calcsize(fmt)
        if size > offsetsize or code in TIFF.TAG_READERS:
            self.valueoffset = offset = unpack(parent.offsetformat, value)[0]
            if offset < 8 or offset > fh.size - size:
                raise TiffTag.Error('invalid tag value offset')
            fh.seek(offset)
            if code in TIFF.TAG_READERS:
                readfunc = TIFF.TAG_READERS[code]
                value = readfunc(fh, byteorder, dtype, count, offsetsize)
            elif type_ == 7 or (count > 1 and dtype[-1] == 'B'):
                value = read_bytes(fh, byteorder, dtype, count, offsetsize)
            elif code in TIFF.TAGS or dtype[-1] == 's':
                value = unpack(fmt, fh.read(size))
            else:
                value = read_numpy(fh, byteorder, dtype, count, offsetsize)
        elif dtype[-1] == 'B' or type_ == 7:
            value = value[:size]
        else:
            value = unpack(fmt, value[:size])
        process = code not in TIFF.TAG_READERS and code not in TIFF.TAG_TUPLE and (type_ != 7)
        if process and dtype[-1] == 's' and isinstance(value[0], bytes):
            value = value[0]
            try:
                value = bytes2str(stripascii(value).strip())
            except UnicodeDecodeError:
                warnings.warn('tag %i: coercing invalid ASCII to bytes' % code)
                dtype = '1B'
        else:
            if code in TIFF.TAG_ENUM:
                t = TIFF.TAG_ENUM[code]
                try:
                    value = tuple((t(v) for v in value))
                except ValueError as e:
                    warnings.warn(str(e))
            if process:
                if len(value) == 1:
                    value = value[0]
        self.code = code
        self.dtype = dtype
        self.count = count
        self.value = value

    @property
    def name(self):
        return TIFF.TAGS.get(self.code, str(self.code))

    def _fix_lsm_bitspersample(self, parent):
        """Correct LSM bitspersample tag.

        Old LSM writers may use a separate region for two 16-bit values,
        although they fit into the tag value element of the tag.

        """
        if self.code == 258 and self.count == 2:
            warnings.warn('correcting LSM bitspersample tag')
            tof = parent.offsetformat[parent.offsetsize]
            self.valueoffset = struct.unpack(tof, self._value)[0]
            parent.filehandle.seek(self.valueoffset)
            self.value = struct.unpack('<HH', parent.filehandle.read(4))

    def __str__(self, detail=0, width=79):
        """Return string containing information about tag."""
        height = 1 if detail <= 0 else 8 * detail
        tcode = '%i%s' % (self.count * int(self.dtype[0]), self.dtype[1])
        line = 'TiffTag %i %s  %s @%i  ' % (self.code, self.name, tcode, self.valueoffset)[:width]
        if self.code in TIFF.TAG_ENUM:
            if self.count == 1:
                value = TIFF.TAG_ENUM[self.code](self.value).name
            else:
                value = pformat(tuple((v.name for v in self.value)))
        else:
            value = pformat(self.value, width=width, height=height)
        if detail <= 0:
            line += value
            line = line[:width]
        else:
            line += '\n' + value
        return line