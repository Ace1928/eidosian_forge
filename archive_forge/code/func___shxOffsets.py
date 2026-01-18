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
def __shxOffsets(self):
    """Reads the shape offset positions from a .shx file"""
    shx = self.shx
    if not shx:
        raise ShapefileException('Shapefile Reader requires a shapefile or file-like object. (no shx file found')
    shx.seek(100)
    shxRecords = _Array('i', shx.read(2 * self.numShapes * 4))
    if sys.byteorder != 'big':
        shxRecords.byteswap()
    self._offsets = [2 * el for el in shxRecords[::2]]