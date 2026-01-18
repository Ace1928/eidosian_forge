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
def _is_writable(keyframe: TiffPage) -> bool:
    """Return True if chunks are writable."""
    return keyframe.compression == 1 and keyframe.fillorder == 1 and (keyframe.sampleformat in {1, 2, 3, 6}) and (keyframe.bitspersample in {8, 16, 32, 64, 128})