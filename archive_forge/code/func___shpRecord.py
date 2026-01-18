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
def __shpRecord(self, s):
    f = self.__getFileObj(self.shp)
    offset = f.tell()
    self.shpNum += 1
    f.write(pack('>2i', self.shpNum, 0))
    start = f.tell()
    if self.shapeType is None and s.shapeType != NULL:
        self.shapeType = s.shapeType
    if s.shapeType != NULL and s.shapeType != self.shapeType:
        raise Exception("The shape's type (%s) must match the type of the shapefile (%s)." % (s.shapeType, self.shapeType))
    f.write(pack('<i', s.shapeType))
    if s.shapeType in (1, 11, 21):
        self.__bbox(s)
    if s.shapeType in (3, 5, 8, 13, 15, 18, 23, 25, 28, 31):
        try:
            f.write(pack('<4d', *self.__bbox(s)))
        except error:
            raise ShapefileException('Failed to write bounding box for record %s. Expected floats.' % self.shpNum)
    if s.shapeType in (3, 5, 13, 15, 23, 25, 31):
        f.write(pack('<i', len(s.parts)))
    if s.shapeType in (3, 5, 8, 13, 15, 18, 23, 25, 28, 31):
        f.write(pack('<i', len(s.points)))
    if s.shapeType in (3, 5, 13, 15, 23, 25, 31):
        for p in s.parts:
            f.write(pack('<i', p))
    if s.shapeType == 31:
        for pt in s.partTypes:
            f.write(pack('<i', pt))
    if s.shapeType in (3, 5, 8, 13, 15, 18, 23, 25, 28, 31):
        try:
            [f.write(pack('<2d', *p[:2])) for p in s.points]
        except error:
            raise ShapefileException('Failed to write points for record %s. Expected floats.' % self.shpNum)
    if s.shapeType in (13, 15, 18, 31):
        try:
            f.write(pack('<2d', *self.__zbox(s)))
        except error:
            raise ShapefileException('Failed to write elevation extremes for record %s. Expected floats.' % self.shpNum)
        try:
            if hasattr(s, 'z'):
                f.write(pack('<%sd' % len(s.z), *s.z))
            else:
                [f.write(pack('<d', p[2] if len(p) > 2 else 0)) for p in s.points]
        except error:
            raise ShapefileException('Failed to write elevation values for record %s. Expected floats.' % self.shpNum)
    if s.shapeType in (13, 15, 18, 23, 25, 28, 31):
        try:
            f.write(pack('<2d', *self.__mbox(s)))
        except error:
            raise ShapefileException('Failed to write measure extremes for record %s. Expected floats' % self.shpNum)
        try:
            if hasattr(s, 'm'):
                f.write(pack('<%sd' % len(s.m), *[m if m is not None else NODATA for m in s.m]))
            else:
                mpos = 3 if s.shapeType in (13, 15, 18, 31) else 2
                [f.write(pack('<d', p[mpos] if len(p) > mpos and p[mpos] is not None else NODATA)) for p in s.points]
        except error:
            raise ShapefileException('Failed to write measure values for record %s. Expected floats' % self.shpNum)
    if s.shapeType in (1, 11, 21):
        try:
            f.write(pack('<2d', s.points[0][0], s.points[0][1]))
        except error:
            raise ShapefileException('Failed to write point for record %s. Expected floats.' % self.shpNum)
    if s.shapeType == 11:
        self.__zbox(s)
        if hasattr(s, 'z'):
            try:
                if not s.z:
                    s.z = (0,)
                f.write(pack('<d', s.z[0]))
            except error:
                raise ShapefileException('Failed to write elevation value for record %s. Expected floats.' % self.shpNum)
        else:
            try:
                if len(s.points[0]) < 3:
                    s.points[0].append(0)
                f.write(pack('<d', s.points[0][2]))
            except error:
                raise ShapefileException('Failed to write elevation value for record %s. Expected floats.' % self.shpNum)
    if s.shapeType in (11, 21):
        self.__mbox(s)
        if hasattr(s, 'm'):
            try:
                if not s.m or s.m[0] is None:
                    s.m = (NODATA,)
                f.write(pack('<1d', s.m[0]))
            except error:
                raise ShapefileException('Failed to write measure value for record %s. Expected floats.' % self.shpNum)
        else:
            try:
                mpos = 3 if s.shapeType == 11 else 2
                if len(s.points[0]) < mpos + 1:
                    s.points[0].append(NODATA)
                elif s.points[0][mpos] is None:
                    s.points[0][mpos] = NODATA
                f.write(pack('<1d', s.points[0][mpos]))
            except error:
                raise ShapefileException('Failed to write measure value for record %s. Expected floats.' % self.shpNum)
    finish = f.tell()
    length = (finish - start) // 2
    f.seek(start - 4)
    f.write(pack('>i', length))
    f.seek(finish)
    return (offset, length)