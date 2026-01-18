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
def _series_fluoview(self) -> list[TiffPageSeries] | None:
    """Return image series in FluoView file."""
    meta = self.fluoview_metadata
    if meta is None:
        return None
    pages = self.pages._getlist(validate=False)
    mmhd = list(reversed(meta['Dimensions']))
    axes = ''.join((TIFF.MM_DIMENSIONS.get(i[0].upper(), 'Q') for i in mmhd))
    shape = tuple((int(i[1]) for i in mmhd))
    self.is_uniform = True
    return [TiffPageSeries(pages, shape, pages[0].dtype, axes, name=meta['ImageName'], kind='fluoview')]