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
def _dtype_str(dtype: numpy.dtype[Any], /) -> str:
    """Return dtype as string with native byte order."""
    if dtype.itemsize == 1:
        byteorder = '|'
    else:
        byteorder = {'big': '>', 'little': '<'}[sys.byteorder]
    return byteorder + dtype.str[1:]