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
def _shaped_series(self):
    """Return image series in "shaped" file."""
    pages = self.pages
    pages.useframes = True
    lenpages = len(pages)

    def append_series(series, pages, axes, shape, reshape, name, truncated):
        page = pages[0]
        if not axes:
            shape = page.shape
            axes = page.axes
            if len(pages) > 1:
                shape = (len(pages),) + shape
                axes = 'Q' + axes
        size = product(shape)
        resize = product(reshape)
        if page.is_contiguous and resize > size and (resize % size == 0):
            if truncated is None:
                truncated = True
            axes = 'Q' + axes
            shape = (resize // size,) + shape
        try:
            axes = reshape_axes(axes, shape, reshape)
            shape = reshape
        except ValueError as e:
            warnings.warn(str(e))
        series.append(TiffPageSeries(pages, shape, page.dtype, axes, name=name, stype='Shaped', truncated=truncated))
    keyframe = axes = shape = reshape = name = None
    series = []
    index = 0
    while True:
        if index >= lenpages:
            break
        pages.keyframe = index
        keyframe = pages[index]
        if not keyframe.is_shaped:
            warnings.warn('invalid shape metadata or corrupted file')
            return
        axes = None
        shape = None
        metadata = json_description_metadata(keyframe.is_shaped)
        name = metadata.get('name', '')
        reshape = metadata['shape']
        truncated = metadata.get('truncated', None)
        if 'axes' in metadata:
            axes = metadata['axes']
            if len(axes) == len(reshape):
                shape = reshape
            else:
                axes = ''
                warnings.warn('axes do not match shape')
        spages = [keyframe]
        size = product(reshape)
        npages, mod = divmod(size, product(keyframe.shape))
        if mod:
            warnings.warn('series shape does not match page shape')
            return
        if 1 < npages <= lenpages - index:
            size *= keyframe._dtype.itemsize
            if truncated:
                npages = 1
            elif keyframe.is_final and keyframe.offset + size < pages[index + 1].offset:
                truncated = False
            else:
                truncated = False
                for j in range(index + 1, index + npages):
                    page = pages[j]
                    page.keyframe = keyframe
                    spages.append(page)
        append_series(series, spages, axes, shape, reshape, name, truncated)
        index += npages
    return series