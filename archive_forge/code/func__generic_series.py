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
def _generic_series(self):
    """Return image series in file."""
    if self.pages.useframes:
        page = self.pages[0]
        shape = page.shape
        axes = page.axes
        if len(self.pages) > 1:
            shape = (len(self.pages),) + shape
            axes = 'I' + axes
        return [TiffPageSeries(self.pages[:], shape, page.dtype, axes, stype='movie')]
    self.pages.clear(False)
    self.pages.load()
    result = []
    keys = []
    series = {}
    compressions = TIFF.DECOMPESSORS
    for page in self.pages:
        if not page.shape:
            continue
        key = page.shape + (page.axes, page.compression in compressions)
        if key in series:
            series[key].append(page)
        else:
            keys.append(key)
            series[key] = [page]
    for key in keys:
        pages = series[key]
        page = pages[0]
        shape = page.shape
        axes = page.axes
        if len(pages) > 1:
            shape = (len(pages),) + shape
            axes = 'I' + axes
        result.append(TiffPageSeries(pages, shape, page.dtype, axes, stype='Generic'))
    return result