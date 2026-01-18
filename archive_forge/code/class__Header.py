import sys
import io
import copy
import array
import itertools
import struct
import zlib
from collections import namedtuple
from io import BytesIO
import numpy as np
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
class _Header:
    __slots__ = ('byteorder', 'zoomLevels', 'chromosomeTreeOffset', 'fullDataOffset', 'fullIndexOffset', 'fieldCount', 'definedFieldCount', 'autoSqlOffset', 'totalSummaryOffset', 'uncompressBufSize', 'extraIndicesOffset')
    formatter = struct.Struct('=IHHQQQHHQQIQ')
    size = formatter.size
    signature = 2273964779
    bbiCurrentVersion = 4

    @classmethod
    def fromfile(cls, stream):
        magic = stream.read(4)
        if int.from_bytes(magic, byteorder='little') == _Header.signature:
            byteorder = '<'
        elif int.from_bytes(magic, byteorder='big') == _Header.signature:
            byteorder = '>'
        else:
            raise ValueError('not a bigBed file')
        formatter = struct.Struct(byteorder + 'HHQQQHHQQIQ')
        header = _Header()
        header.byteorder = byteorder
        size = formatter.size
        data = stream.read(size)
        version, header.zoomLevels, header.chromosomeTreeOffset, header.fullDataOffset, header.fullIndexOffset, header.fieldCount, header.definedFieldCount, header.autoSqlOffset, header.totalSummaryOffset, header.uncompressBufSize, header.extraIndicesOffset = formatter.unpack(data)
        assert version == _Header.bbiCurrentVersion
        definedFieldCount = header.definedFieldCount
        if definedFieldCount < 3 or definedFieldCount > 12:
            raise ValueError('expected between 3 and 12 columns, found %d' % definedFieldCount)
        return header

    def __bytes__(self):
        return _Header.formatter.pack(_Header.signature, _Header.bbiCurrentVersion, self.zoomLevels, self.chromosomeTreeOffset, self.fullDataOffset, self.fullIndexOffset, self.fieldCount, self.definedFieldCount, self.autoSqlOffset, self.totalSummaryOffset, self.uncompressBufSize, self.extraIndicesOffset)