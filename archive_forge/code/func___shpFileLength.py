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
def __shpFileLength(self):
    """Calculates the file length of the shp file."""
    start = self.shp.tell()
    self.shp.seek(0, 2)
    size = self.shp.tell()
    size //= 2
    self.shp.seek(start)
    return size