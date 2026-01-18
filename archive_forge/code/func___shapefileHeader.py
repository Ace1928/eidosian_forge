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
def __shapefileHeader(self, fileObj, headerType='shp'):
    """Writes the specified header type to the specified file-like object.
        Several of the shapefile formats are so similar that a single generic
        method to read or write them is warranted."""
    f = self.__getFileObj(fileObj)
    f.seek(0)
    f.write(pack('>6i', 9994, 0, 0, 0, 0, 0))
    if headerType == 'shp':
        f.write(pack('>i', self.__shpFileLength()))
    elif headerType == 'shx':
        f.write(pack('>i', (100 + self.shpNum * 8) // 2))
    if self.shapeType is None:
        self.shapeType = NULL
    f.write(pack('<2i', 1000, self.shapeType))
    if self.shapeType != 0:
        try:
            bbox = self.bbox()
            if bbox is None:
                bbox = [0, 0, 0, 0]
            f.write(pack('<4d', *bbox))
        except error:
            raise ShapefileException('Failed to write shapefile bounding box. Floats required.')
    else:
        f.write(pack('<4d', 0, 0, 0, 0))
    if self.shapeType in (11, 13, 15, 18):
        zbox = self.zbox()
        if zbox is None:
            zbox = [0, 0]
    else:
        zbox = [0, 0]
    if self.shapeType in (11, 13, 15, 18, 21, 23, 25, 28, 31):
        mbox = self.mbox()
        if mbox is None:
            mbox = [0, 0]
    else:
        mbox = [0, 0]
    try:
        f.write(pack('<4d', zbox[0], zbox[1], mbox[0], mbox[1]))
    except error:
        raise ShapefileException('Failed to write shapefile elevation and measure values. Floats required.')