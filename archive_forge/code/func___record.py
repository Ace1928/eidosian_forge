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
def __record(self, fieldTuples, recLookup, recStruct, oid=None):
    """Reads and returns a dbf record row as a list of values. Requires specifying
        a list of field info tuples 'fieldTuples', a record name-index dict 'recLookup', 
        and a Struct instance 'recStruct' for unpacking these fields. 
        """
    f = self.__getFileObj(self.dbf)
    recordContents = recStruct.unpack(f.read(recStruct.size))
    if recordContents[0] != b' ':
        return None
    recordContents = recordContents[1:]
    if len(fieldTuples) != len(recordContents):
        raise ShapefileException('Number of record values ({}) is different from the requested                             number of fields ({})'.format(len(recordContents), len(fieldTuples)))
    record = []
    for (name, typ, size, deci), value in zip(fieldTuples, recordContents):
        if typ in ('N', 'F'):
            value = value.split(b'\x00')[0]
            value = value.replace(b'*', b'')
            if value == b'':
                value = None
            elif deci:
                try:
                    value = float(value)
                except ValueError:
                    value = None
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = int(float(value))
                    except ValueError:
                        value = None
        elif typ == 'D':
            if not value.replace(b'\x00', b'').replace(b' ', b'').replace(b'0', b''):
                value = None
            else:
                try:
                    y, m, d = (int(value[:4]), int(value[4:6]), int(value[6:8]))
                    value = date(y, m, d)
                except:
                    value = u(value.strip())
        elif typ == 'L':
            if value == b' ':
                value = None
            elif value in b'YyTt1':
                value = True
            elif value in b'NnFf0':
                value = False
            else:
                value = None
        else:
            value = u(value, self.encoding, self.encodingErrors)
            value = value.strip().rstrip('\x00')
        record.append(value)
    return _Record(recLookup, record, oid)