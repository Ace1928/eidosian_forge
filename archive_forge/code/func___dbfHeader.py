from struct import pack, unpack, calcsize, error, Struct
import os
import sys
import time
import array
import tempfile
import logging
import io
from datetime import date
import zipfile
def __dbfHeader(self):
    """Writes the dbf header and field descriptors."""
    f = self.__getFileObj(self.dbf)
    f.seek(0)
    version = 3
    year, month, day = time.localtime()[:3]
    year -= 1900
    fields = [field for field in self.fields if field[0] != 'DeletionFlag']
    if not fields:
        raise ShapefileException('Shapefile dbf file must contain at least one field.')
    numRecs = self.recNum
    numFields = len(fields)
    headerLength = numFields * 32 + 33
    if headerLength >= 65535:
        raise ShapefileException('Shapefile dbf header length exceeds maximum length.')
    recordLength = sum([int(field[2]) for field in fields]) + 1
    header = pack('<BBBBLHH20x', version, year, month, day, numRecs, headerLength, recordLength)
    f.write(header)
    for field in fields:
        name, fieldType, size, decimal = field
        name = b(name, self.encoding, self.encodingErrors)
        name = name.replace(b' ', b'_')
        name = name[:10].ljust(11).replace(b' ', b'\x00')
        fieldType = b(fieldType, 'ascii')
        size = int(size)
        fld = pack('<11sc4xBB14x', name, fieldType, size, decimal)
        f.write(fld)
    f.write(b'\r')