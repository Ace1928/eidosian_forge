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
def _series_sis(self) -> list[TiffPageSeries] | None:
    """Return image series in Olympus SIS file."""
    meta = self.sis_metadata
    if meta is None:
        return None
    pages = self.pages._getlist(validate=False)
    page = pages[0]
    lenpages = len(pages)
    if 'shape' in meta and 'axes' in meta:
        shape = meta['shape'] + page.shape
        axes = meta['axes'] + page.axes
    else:
        shape = (lenpages,) + page.shape
        axes = 'I' + page.axes
    self.is_uniform = True
    return [TiffPageSeries(pages, shape, page.dtype, axes, kind='sis')]