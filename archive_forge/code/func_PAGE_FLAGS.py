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
@cached_property
def PAGE_FLAGS(self) -> set[str]:
    exclude = {'reduced', 'mask', 'final', 'memmappable', 'contiguous', 'tiled', 'subsampled', 'jfif'}
    return {a[3:] for a in dir(TiffPage) if a[:3] == 'is_' and a[3:] not in exclude}