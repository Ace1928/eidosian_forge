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
def _lsm_series(self):
    """Return main image series in LSM file. Skip thumbnails."""
    lsmi = self.lsm_metadata
    axes = TIFF.CZ_LSMINFO_SCANTYPE[lsmi['ScanType']]
    if self.pages[0].photometric == 2:
        axes = axes.replace('C', '').replace('XY', 'XYC')
    if lsmi.get('DimensionP', 0) > 1:
        axes += 'P'
    if lsmi.get('DimensionM', 0) > 1:
        axes += 'M'
    axes = axes[::-1]
    shape = tuple((int(lsmi[TIFF.CZ_LSMINFO_DIMENSIONS[i]]) for i in axes))
    name = lsmi.get('Name', '')
    self.pages.keyframe = 0
    pages = self.pages[::2]
    dtype = pages[0].dtype
    series = [TiffPageSeries(pages, shape, dtype, axes, name=name, stype='LSM')]
    if self.pages[1].is_reduced:
        self.pages.keyframe = 1
        pages = self.pages[1::2]
        dtype = pages[0].dtype
        cp, i = (1, 0)
        while cp < len(pages) and i < len(shape) - 2:
            cp *= shape[i]
            i += 1
        shape = shape[:i] + pages[0].shape
        axes = axes[:i] + 'CYX'
        series.append(TiffPageSeries(pages, shape, dtype, axes, name=name, stype='LSMreduced'))
    return series