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
class CHUNKMODE(enum.IntEnum):
    """ZarrStore chunk modes.

    Specifies how to chunk data in Zarr stores.

    """
    STRILE = 0
    'Chunk is strip or tile.'
    PLANE = 1
    'Chunk is image plane.'
    PAGE = 2
    'Chunk is image in page.'
    FILE = 3
    'Chunk is image in file.'