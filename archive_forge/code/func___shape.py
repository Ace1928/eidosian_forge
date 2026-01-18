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
def __shape(self, oid=None, bbox=None):
    """Returns the header info and geometry for a single shape."""
    f = self.__getFileObj(self.shp)
    record = Shape(oid=oid)
    nParts = nPoints = zmin = zmax = mmin = mmax = None
    recNum, recLength = unpack('>2i', f.read(8))
    next = f.tell() + 2 * recLength
    shapeType = unpack('<i', f.read(4))[0]
    record.shapeType = shapeType
    if shapeType == 0:
        record.points = []
    elif shapeType in (3, 5, 8, 13, 15, 18, 23, 25, 28, 31):
        record.bbox = _Array('d', unpack('<4d', f.read(32)))
        if bbox is not None and (not bbox_overlap(bbox, record.bbox)):
            f.seek(next)
            return None
    if shapeType in (3, 5, 13, 15, 23, 25, 31):
        nParts = unpack('<i', f.read(4))[0]
    if shapeType in (3, 5, 8, 13, 15, 18, 23, 25, 28, 31):
        nPoints = unpack('<i', f.read(4))[0]
    if nParts:
        record.parts = _Array('i', unpack('<%si' % nParts, f.read(nParts * 4)))
    if shapeType == 31:
        record.partTypes = _Array('i', unpack('<%si' % nParts, f.read(nParts * 4)))
    if nPoints:
        flat = unpack('<%sd' % (2 * nPoints), f.read(16 * nPoints))
        record.points = list(izip(*(iter(flat),) * 2))
    if shapeType in (13, 15, 18, 31):
        zmin, zmax = unpack('<2d', f.read(16))
        record.z = _Array('d', unpack('<%sd' % nPoints, f.read(nPoints * 8)))
    if shapeType in (13, 15, 18, 23, 25, 28, 31):
        if next - f.tell() >= 16:
            mmin, mmax = unpack('<2d', f.read(16))
        if next - f.tell() >= nPoints * 8:
            record.m = []
            for m in _Array('d', unpack('<%sd' % nPoints, f.read(nPoints * 8))):
                if m > NODATA:
                    record.m.append(m)
                else:
                    record.m.append(None)
        else:
            record.m = [None for _ in range(nPoints)]
    if shapeType in (1, 11, 21):
        record.points = [_Array('d', unpack('<2d', f.read(16)))]
    if shapeType == 11:
        record.z = list(unpack('<d', f.read(8)))
    if shapeType in (21, 11):
        if next - f.tell() >= 8:
            m, = unpack('<d', f.read(8))
        else:
            m = NODATA
        if m > NODATA:
            record.m = [m]
        else:
            record.m = [None]
    f.seek(next)
    return record