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
def _ndindex(shape: tuple[int, ...], chunks: tuple[int, ...], /) -> Iterator[str]:
    """Return iterator over all chunk index strings."""
    assert len(shape) == len(chunks)
    chunked = tuple((i // j + (1 if i % j else 0) for i, j in zip(shape, chunks)))
    for indices in numpy.ndindex(chunked):
        yield '.'.join((str(index) for index in indices))