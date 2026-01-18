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
def __shpHeader(self):
    """Reads the header information from a .shp file."""
    if not self.shp:
        raise ShapefileException('Shapefile Reader requires a shapefile or file-like object. (no shp file found')
    shp = self.shp
    shp.seek(24)
    self.shpLength = unpack('>i', shp.read(4))[0] * 2
    shp.seek(32)
    self.shapeType = unpack('<i', shp.read(4))[0]
    self.bbox = _Array('d', unpack('<4d', shp.read(32)))
    self.zbox = _Array('d', unpack('<2d', shp.read(16)))
    self.mbox = []
    for m in _Array('d', unpack('<2d', shp.read(16))):
        if m > NODATA:
            self.mbox.append(m)
        else:
            self.mbox.append(None)