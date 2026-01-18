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
class TiffPageSeries(object):
    """Series of TIFF pages with compatible shape and data type.

    Attributes
    ----------
    pages : list of TiffPage
        Sequence of TiffPages in series.
    dtype : numpy.dtype
        Data type (native byte order) of the image array in series.
    shape : tuple
        Dimensions of the image array in series.
    axes : str
        Labels of axes in shape. See TiffPage.axes.
    offset : int or None
        Position of image data in file if memory-mappable, else None.

    """

    def __init__(self, pages, shape, dtype, axes, parent=None, name=None, transform=None, stype=None, truncated=False):
        """Initialize instance."""
        self.index = 0
        self._pages = pages
        self.shape = tuple(shape)
        self.axes = ''.join(axes)
        self.dtype = numpy.dtype(dtype)
        self.stype = stype if stype else ''
        self.name = name if name else ''
        self.transform = transform
        if parent:
            self.parent = parent
        elif pages:
            self.parent = pages[0].parent
        else:
            self.parent = None
        if len(pages) == 1 and (not truncated):
            self._len = int(product(self.shape) // product(pages[0].shape))
        else:
            self._len = len(pages)

    def asarray(self, out=None):
        """Return image data from series of TIFF pages as numpy array."""
        if self.parent:
            result = self.parent.asarray(series=self, out=out)
            if self.transform is not None:
                result = self.transform(result)
            return result

    @lazyattr
    def offset(self):
        """Return offset to series data in file, if any."""
        if not self._pages:
            return
        pos = 0
        for page in self._pages:
            if page is None:
                return
            if not page.is_final:
                return
            if not pos:
                pos = page.is_contiguous[0] + page.is_contiguous[1]
                continue
            if pos != page.is_contiguous[0]:
                return
            pos += page.is_contiguous[1]
        page = self._pages[0]
        offset = page.is_contiguous[0]
        if (page.is_imagej or page.is_shaped) and len(self._pages) == 1:
            return offset
        if pos == offset + product(self.shape) * self.dtype.itemsize:
            return offset

    @property
    def ndim(self):
        """Return number of array dimensions."""
        return len(self.shape)

    @property
    def size(self):
        """Return number of elements in array."""
        return int(product(self.shape))

    @property
    def pages(self):
        """Return sequence of all pages in series."""
        return self

    def __len__(self):
        """Return number of TiffPages in series."""
        return self._len

    def __getitem__(self, key):
        """Return specified TiffPage."""
        if len(self._pages) == 1 and 0 < key < self._len:
            index = self._pages[0].index
            return self.parent.pages[index + key]
        return self._pages[key]

    def __iter__(self):
        """Return iterator over TiffPages in series."""
        if len(self._pages) == self._len:
            for page in self._pages:
                yield page
        else:
            pages = self.parent.pages
            index = self._pages[0].index
            for i in range(self._len):
                yield pages[index + i]

    def __str__(self):
        """Return string with information about series."""
        s = '  '.join((s for s in (snipstr("'%s'" % self.name, 20) if self.name else '', 'x'.join((str(i) for i in self.shape)), str(self.dtype), self.axes, self.stype, '%i Pages' % len(self.pages), 'Offset=%i' % self.offset if self.offset else '') if s))
        return 'TiffPageSeries %i  %s' % (self.index, s)