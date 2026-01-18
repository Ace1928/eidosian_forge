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
class DATATYPE(enum.IntEnum):
    """TIFF tag data types."""
    BYTE = 1
    '8-bit unsigned integer.'
    ASCII = 2
    '8-bit byte with last byte null, containing 7-bit ASCII code.'
    SHORT = 3
    '16-bit unsigned integer.'
    LONG = 4
    '32-bit unsigned integer.'
    RATIONAL = 5
    'Two 32-bit unsigned integers, numerator and denominator of fraction.'
    SBYTE = 6
    '8-bit signed integer.'
    UNDEFINED = 7
    '8-bit byte that may contain anything.'
    SSHORT = 8
    '16-bit signed integer.'
    SLONG = 9
    '32-bit signed integer.'
    SRATIONAL = 10
    'Two 32-bit signed integers, numerator and denominator of fraction.'
    FLOAT = 11
    'Single precision (4-byte) IEEE format.'
    DOUBLE = 12
    'Double precision (8-byte) IEEE format.'
    IFD = 13
    'Unsigned 4 byte IFD offset.'
    UNICODE = 14
    COMPLEX = 15
    LONG8 = 16
    'Unsigned 8 byte integer (BigTIFF).'
    SLONG8 = 17
    'Signed 8 byte integer (BigTIFF).'
    IFD8 = 18
    'Unsigned 8 byte IFD offset (BigTIFF).'