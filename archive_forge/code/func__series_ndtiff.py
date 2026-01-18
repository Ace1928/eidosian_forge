from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
def _series_ndtiff(self) -> list[TiffPageSeries] | None:
    """Return series in NDTiff v2 and v3 files."""
    if not self.filehandle.is_file:
        logger().warning(f'{self!r} NDTiff.index not found for {self.filehandle._fh!r}')
        return None
    indexfile = os.path.join(self.filehandle.dirname, 'NDTiff.index')
    if not os.path.exists(indexfile):
        logger().warning(f'{self!r} NDTiff.index not found')
        return None
    keyframes: dict[str, TiffPage] = {}
    shape: tuple[int, ...]
    dims: tuple[str, ...]
    page: TiffPage | TiffFrame
    pageindex = 0
    pixel_types = {0: ('uint8', 8), 1: ('uint16', 16), 2: ('uint8', 8), 3: ('uint16', 10), 4: ('uint16', 12), 5: ('uint16', 14), 6: ('uint16', 11)}
    indices: dict[tuple[int, ...], TiffPage | TiffFrame] = {}
    categories: dict[str, dict[str, int]] = {}
    first = True
    for axes_dict, filename, dataoffset, width, height, pixeltype, compression, metaoffset, metabytecount, metacompression in read_ndtiff_index(indexfile):
        if filename in keyframes:
            pageindex += 1
            keyframe = keyframes[filename]
            page = TiffFrame(keyframe.parent, pageindex, offset=None, keyframe=keyframe, dataoffsets=(dataoffset,), databytecounts=keyframe.databytecounts)
            if page.shape[:2] != (height, width):
                raise ValueError(f'NDTiff.index does not match TIFF shape {page.shape[:2]} != {(height, width)}')
            if compression != 0:
                raise ValueError('NDTiff.index compression {compression} not supported')
            if page.compression != 1:
                raise ValueError(f'NDTiff.index does not match TIFF compression {page.compression!r}')
            if pixeltype not in pixel_types:
                raise ValueError(f'NDTiff.index unknown pixel type {pixeltype}')
            dtype, _ = pixel_types[pixeltype]
            if page.dtype != dtype:
                raise ValueError(f'NDTiff.index pixeltype does not match TIFF dtype {page.dtype} != {dtype}')
        elif filename == self.filename:
            pageindex = 0
            page = self.pages.first
            keyframes[filename] = page
        else:
            pageindex = 0
            with TiffFile(os.path.join(self.filehandle.dirname, filename)) as tif:
                page = tif.pages.first
            keyframes[filename] = page
        index: int | str
        if first:
            for axis, index in axes_dict.items():
                if isinstance(index, str):
                    categories[axis] = {index: 0}
                    axes_dict[axis] = 0
            first = False
        elif categories:
            for axis, values in categories.items():
                index = axes_dict[axis]
                assert isinstance(index, str)
                if index not in values:
                    values[index] = max(values.values()) + 1
                axes_dict[axis] = values[index]
        indices[tuple(axes_dict.values())] = page
        dims = tuple(axes_dict.keys())
    indices_array = numpy.array(list(indices.keys()), dtype=numpy.int32)
    min_index = numpy.min(indices_array, axis=0).tolist()
    max_index = numpy.max(indices_array, axis=0).tolist()
    shape = tuple((j - i + 1 for i, j in zip(min_index, max_index)))
    order = order_axes(indices_array, squeeze=False)
    shape = tuple((shape[i] for i in order))
    dims = tuple((dims[i] for i in order))
    indices = {tuple((index[i] - min_index[i] for i in order)): value for index, value in indices.items()}
    pages: list[TiffPage | TiffFrame | None] = []
    for idx in numpy.ndindex(shape):
        pages.append(indices.get(idx, None))
    keyframe = next((i for i in keyframes.values()))
    shape += keyframe.shape
    dims += keyframe.dims
    axes = ''.join((TIFF.AXES_CODES.get(i.lower(), 'Q') for i in dims))
    self.is_uniform = True
    return [TiffPageSeries(pages, shape=shape, dtype=keyframe.dtype, axes=axes, parent=self, kind='ndtiff', multifile=len(keyframes) > 1, squeeze=True)]