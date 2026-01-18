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
def __recordFmt(self, fields=None):
    """Calculates the format and size of a .dbf record. Optional 'fields' arg 
        specifies which fieldnames to unpack and which to ignore. Note that this
        always includes the DeletionFlag at index 0, regardless of the 'fields' arg. 
        """
    if self.numRecords is None:
        self.__dbfHeader()
    structcodes = ['%ds' % fieldinfo[2] for fieldinfo in self.fields]
    if fields is not None:
        structcodes = [code if fieldinfo[0] in fields or fieldinfo[0] == 'DeletionFlag' else '%dx' % fieldinfo[2] for fieldinfo, code in zip(self.fields, structcodes)]
    fmt = ''.join(structcodes)
    fmtSize = calcsize(fmt)
    while fmtSize < self.__recordLength:
        fmt += 'x'
        fmtSize += 1
    return (fmt, fmtSize)