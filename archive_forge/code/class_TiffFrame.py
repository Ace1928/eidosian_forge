from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
class TiffFrame(object):
    """Lightweight TIFF image file directory (IFD).

    Only a limited number of tag values are read from file, e.g. StripOffsets,
    and StripByteCounts. Other tag values are assumed to be identical with a
    specified TiffPage instance, the keyframe.

    TiffFrame is intended to reduce resource usage and speed up reading data
    from file, not for introspection of metadata.

    Not compatible with Python 2.

    """
    __slots__ = ('keyframe', 'parent', 'index', 'offset', 'dataoffsets', 'databytecounts')
    is_mdgel = False
    tags = {}

    def __init__(self, parent, index, keyframe):
        """Read specified tags from file.

        The file handle position must be at the offset to a valid IFD.

        """
        self.keyframe = keyframe
        self.parent = parent
        self.index = index
        self.dataoffsets = None
        self.databytecounts = None
        unpack = struct.unpack
        fh = parent.filehandle
        self.offset = fh.tell()
        try:
            tagno = unpack(parent.tagnoformat, fh.read(parent.tagnosize))[0]
            if tagno > 4096:
                raise ValueError('suspicious number of tags')
        except Exception:
            raise ValueError('corrupted page list at offset %i' % self.offset)
        tagcodes = {273, 279, 324, 325}
        tagsize = parent.tagsize
        codeformat = parent.tagformat1[:2]
        data = fh.read(tagsize * tagno)
        index = -tagsize
        for _ in range(tagno):
            index += tagsize
            code = unpack(codeformat, data[index:index + 2])[0]
            if code not in tagcodes:
                continue
            try:
                tag = TiffTag(parent, data[index:index + tagsize])
            except TiffTag.Error as e:
                warnings.warn(str(e))
                continue
            if code == 273 or code == 324:
                setattr(self, 'dataoffsets', tag.value)
            elif code == 279 or code == 325:
                setattr(self, 'databytecounts', tag.value)

    def aspage(self):
        """Return TiffPage from file."""
        self.parent.filehandle.seek(self.offset)
        return TiffPage(self.parent, index=self.index, keyframe=None)

    def asarray(self, *args, **kwargs):
        """Read image data from file and return as numpy array."""
        kwargs['validate'] = False
        return TiffPage.asarray(self, *args, **kwargs)

    def asrgb(self, *args, **kwargs):
        """Read image data from file and return RGB image as numpy array."""
        kwargs['validate'] = False
        return TiffPage.asrgb(self, *args, **kwargs)

    @property
    def offsets_bytecounts(self):
        """Return simplified offsets and bytecounts."""
        if self.keyframe.is_contiguous:
            return (self.dataoffsets[:1], self.keyframe.is_contiguous[1:])
        return clean_offsets_counts(self.dataoffsets, self.databytecounts)

    @property
    def is_contiguous(self):
        """Return offset and size of contiguous data, else None."""
        if self.keyframe.is_contiguous:
            return (self.dataoffsets[0], self.keyframe.is_contiguous[1])

    @property
    def is_memmappable(self):
        """Return if page's image data in file can be memory-mapped."""
        return self.keyframe.is_memmappable

    def __getattr__(self, name):
        """Return attribute from keyframe."""
        if name in TIFF.FRAME_ATTRS:
            return getattr(self.keyframe, name)
        raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))

    def __str__(self, detail=0):
        """Return string containing information about frame."""
        info = '  '.join((s for s in ('x'.join((str(i) for i in self.shape)), str(self.dtype))))
        return 'TiffFrame %i @%i  %s' % (self.index, self.offset, info)