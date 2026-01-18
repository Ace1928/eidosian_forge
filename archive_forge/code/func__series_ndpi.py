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
def _series_ndpi(self) -> list[TiffPageSeries] | None:
    """Return pyramidal image series in NDPI file."""
    series = self._series_generic()
    if series is None:
        return None
    for s in series:
        s.kind = 'ndpi'
        if s.axes[0] == 'I':
            s._set_dimensions(s.shape, 'Z' + s.axes[1:], None, True)
        if s.is_pyramidal:
            name = s.keyframe.tags.valueof(65427)
            s.name = 'Baseline' if name is None else name
            continue
        mag = s.keyframe.tags.valueof(65421)
        if mag is not None:
            if mag == -1.0:
                s.name = 'Macro'
            elif mag == -2.0:
                s.name = 'Map'
    self.is_uniform = False
    return series