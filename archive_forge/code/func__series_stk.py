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
def _series_stk(self) -> list[TiffPageSeries] | None:
    """Return series in STK file."""
    meta = self.stk_metadata
    if meta is None:
        return None
    page = self.pages.first
    planes = meta['NumberPlanes']
    name = meta.get('Name', '')
    if planes == 1:
        shape = (1,) + page.shape
        axes = 'I' + page.axes
    elif numpy.all(meta['ZDistance'] != 0):
        shape = (planes,) + page.shape
        axes = 'Z' + page.axes
    elif numpy.all(numpy.diff(meta['TimeCreated']) != 0):
        shape = (planes,) + page.shape
        axes = 'T' + page.axes
    else:
        shape = (planes,) + page.shape
        axes = 'I' + page.axes
    self.is_uniform = True
    series = TiffPageSeries([page], shape, page.dtype, axes, name=name, truncated=planes > 1, kind='stk')
    return [series]