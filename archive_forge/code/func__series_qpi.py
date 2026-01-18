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
def _series_qpi(self) -> list[TiffPageSeries] | None:
    """Return image series in PerkinElmer QPI file."""
    series = []
    pages = self.pages
    pages.cache = True
    pages.useframes = False
    pages.set_keyframe(0)
    pages._load()
    page0 = self.pages.first
    ifds = []
    index = 0
    axes = 'C' + page0.axes
    dtype = page0.dtype
    pshape = page0.shape
    while index < len(pages):
        page = pages[index]
        if page.shape != pshape:
            break
        ifds.append(page)
        index += 1
    shape = (len(ifds),) + pshape
    series.append(TiffPageSeries(ifds, shape, dtype, axes, name='Baseline', kind='qpi'))
    if index < len(pages):
        page = pages[index]
        series.append(TiffPageSeries([page], page.shape, page.dtype, page.axes, name='Thumbnail', kind='qpi'))
        index += 1
    if page0.is_tiled:
        while index < len(pages):
            pshape = (pshape[0] // 2, pshape[1] // 2) + pshape[2:]
            ifds = []
            while index < len(pages):
                page = pages[index]
                if page.shape != pshape:
                    break
                ifds.append(page)
                index += 1
            if len(ifds) != len(series[0].pages):
                break
            shape = (len(ifds),) + pshape
            series[0].levels.append(TiffPageSeries(ifds, shape, dtype, axes, name='Resolution', kind='qpi'))
    if series[0].is_pyramidal and index < len(pages):
        page = pages[index]
        series.append(TiffPageSeries([page], page.shape, page.dtype, page.axes, name='Macro', kind='qpi'))
        index += 1
        if index < len(pages):
            page = pages[index]
            series.append(TiffPageSeries([page], page.shape, page.dtype, page.axes, name='Label', kind='qpi'))
    self.is_uniform = False
    return series