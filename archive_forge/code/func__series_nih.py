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
def _series_nih(self) -> list[TiffPageSeries] | None:
    """Return all images in NIH Image file as single series."""
    series = self._series_uniform()
    if series is not None:
        for s in series:
            s.kind = 'nih'
    return series