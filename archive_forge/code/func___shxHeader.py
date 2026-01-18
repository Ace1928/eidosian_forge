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
def __shxHeader(self):
    """Reads the header information from a .shx file."""
    shx = self.shx
    if not shx:
        raise ShapefileException('Shapefile Reader requires a shapefile or file-like object. (no shx file found')
    shx.seek(24)
    shxRecordLength = unpack('>i', shx.read(4))[0] * 2 - 100
    self.numShapes = shxRecordLength // 8