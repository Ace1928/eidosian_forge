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
@staticmethod
def _empty_chunk(shape: tuple[int, ...], dtype: DTypeLike, fillvalue: int | float | None, /) -> NDArray[Any]:
    """Return empty chunk."""
    if fillvalue is None or fillvalue == 0:
        return numpy.zeros(shape, dtype)
    chunk = numpy.empty(shape, dtype)
    chunk[:] = fillvalue
    return chunk