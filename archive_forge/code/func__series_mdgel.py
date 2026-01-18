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
def _series_mdgel(self) -> list[TiffPageSeries] | None:
    """Return image series in MD Gel file."""
    meta = self.mdgel_metadata
    if meta is None:
        return None
    transform: Callable[[NDArray[Any]], NDArray[Any]] | None
    self.pages.useframes = False
    self.pages.set_keyframe(0)
    if meta['FileTag'] in {2, 128}:
        dtype = numpy.dtype('float32')
        scale = meta['ScalePixel']
        scale = scale[0] / scale[1]
        if meta['FileTag'] == 2:

            def transform(a: NDArray[Any], /) -> NDArray[Any]:
                return a.astype('float32') ** 2 * scale
        else:

            def transform(a: NDArray[Any], /) -> NDArray[Any]:
                return a.astype('float32') * scale
    else:
        transform = None
    page = self.pages.first
    self.is_uniform = False
    return [TiffPageSeries([page], page.shape, dtype, page.axes, transform=transform, kind='mdgel')]