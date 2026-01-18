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
def _series_svs(self) -> list[TiffPageSeries] | None:
    """Return image series in Aperio SVS file."""
    if not self.pages.first.is_tiled:
        return None
    series = []
    self.pages.cache = True
    self.pages.useframes = False
    self.pages.set_keyframe(0)
    self.pages._load()
    firstpage = self.pages.first
    if len(self.pages) == 1:
        self.is_uniform = False
        return [TiffPageSeries([firstpage], firstpage.shape, firstpage.dtype, firstpage.axes, name='Baseline', kind='svs')]
    page = self.pages[1]
    thumnail = TiffPageSeries([page], page.shape, page.dtype, page.axes, name='Thumbnail', kind='svs')
    levels = {firstpage.shape: [firstpage]}
    index = 2
    while index < len(self.pages):
        page = cast(TiffPage, self.pages[index])
        if not page.is_tiled or page.is_reduced:
            break
        if page.shape in levels:
            levels[page.shape].append(page)
        else:
            levels[page.shape] = [page]
        index += 1
    zsize = len(levels[firstpage.shape])
    if not all((len(level) == zsize for level in levels.values())):
        logger().warning(f'{self!r} SVS series focal planes do not match')
        zsize = 1
    baseline = TiffPageSeries(levels[firstpage.shape], (zsize,) + firstpage.shape, firstpage.dtype, 'Z' + firstpage.axes, name='Baseline', kind='svs')
    for shape, level in levels.items():
        if shape == firstpage.shape:
            continue
        page = level[0]
        baseline.levels.append(TiffPageSeries(level, (zsize,) + page.shape, page.dtype, 'Z' + page.axes, name='Resolution', kind='svs'))
    series.append(baseline)
    series.append(thumnail)
    for _ in range(2):
        if index == len(self.pages):
            break
        page = self.pages[index]
        if page.subfiletype == 9:
            name = 'Macro'
        else:
            name = 'Label'
        series.append(TiffPageSeries([page], page.shape, page.dtype, page.axes, name=name, kind='svs'))
        index += 1
    self.is_uniform = False
    return series