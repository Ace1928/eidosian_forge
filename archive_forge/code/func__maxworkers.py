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
def _maxworkers(maxworkers: int | None, numchunks: int, chunksize: int, compression: int) -> int:
    """Return number of threads to encode segments."""
    if maxworkers is not None:
        return maxworkers
    if imagecodecs is None or compression <= 1 or numchunks < 2 or (chunksize < 1024) or (compression == 48124):
        return 1
    if chunksize < 131072 and compression in {7, 33007, 32773, 34887}:
        return 1
    if chunksize < 32768 and compression in {5, 8, 32946, 50000, 50013}:
        return 1
    if chunksize < 8192 and compression in {34934, 22610, 34933}:
        return 1
    if chunksize < 2048 and compression in {33003, 33004, 33005, 34712, 50002}:
        return 1
    if chunksize < 1024 and compression in {34925, 50001}:
        return 1
    if compression == 34887:
        return min(numchunks, 4)
    return min(numchunks, TIFF.MAXWORKERS)