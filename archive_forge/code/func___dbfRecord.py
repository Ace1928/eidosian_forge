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
def __dbfRecord(self, record):
    """Writes the dbf records."""
    f = self.__getFileObj(self.dbf)
    if self.recNum == 0:
        self.__dbfHeader()
    f.write(b' ')
    self.recNum += 1
    fields = (field for field in self.fields if field[0] != 'DeletionFlag')
    for (fieldName, fieldType, size, deci), value in zip(fields, record):
        fieldType = fieldType.upper()
        size = int(size)
        if fieldType in ('N', 'F'):
            if value in MISSING:
                value = b'*' * size
            elif not deci:
                try:
                    value = int(value)
                except ValueError:
                    value = int(float(value))
                value = format(value, 'd')[:size].rjust(size)
            else:
                value = float(value)
                value = format(value, '.%sf' % deci)[:size].rjust(size)
        elif fieldType == 'D':
            if isinstance(value, date):
                value = '{:04d}{:02d}{:02d}'.format(value.year, value.month, value.day)
            elif isinstance(value, list) and len(value) == 3:
                value = '{:04d}{:02d}{:02d}'.format(*value)
            elif value in MISSING:
                value = b'0' * 8
            elif is_string(value) and len(value) == 8:
                pass
            else:
                raise ShapefileException('Date values must be either a datetime.date object, a list, a YYYYMMDD string, or a missing value.')
        elif fieldType == 'L':
            if value in MISSING:
                value = b' '
            elif value in [True, 1]:
                value = b'T'
            elif value in [False, 0]:
                value = b'F'
            else:
                value = b' '
        else:
            value = b(value, self.encoding, self.encodingErrors)[:size].ljust(size)
        if not isinstance(value, bytes):
            value = b(value, 'ascii', self.encodingErrors)
        if len(value) != size:
            raise ShapefileException("Shapefile Writer unable to pack incorrect sized value (size %d) into field '%s' (size %d)." % (len(value), fieldName, size))
        f.write(value)