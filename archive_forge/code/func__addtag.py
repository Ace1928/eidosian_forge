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
def _addtag(self, tags: list[tuple[int, bytes, bytes | None, bool]], code: int | str, dtype: int | str, count: int | None, value: Any, writeonce: bool=False, /) -> None:
    """Append (code, ifdentry, ifdvalue, writeonce) to tags list.

        Compute ifdentry and ifdvalue bytes from code, dtype, count, value.

        """
    pack = self._pack
    if not isinstance(code, int):
        code = TIFF.TAGS[code]
    try:
        datatype = cast(int, dtype)
        dataformat = TIFF.DATA_FORMATS[datatype][-1]
    except KeyError as exc:
        try:
            dataformat = cast(str, dtype)
            if dataformat[0] in '<>':
                dataformat = dataformat[1:]
            datatype = TIFF.DATA_DTYPES[dataformat]
        except (KeyError, TypeError):
            raise ValueError(f'unknown dtype {dtype}') from exc
    del dtype
    rawcount = count
    if datatype == 2:
        if isinstance(value, str):
            try:
                value = value.encode('ascii')
            except UnicodeEncodeError as exc:
                raise ValueError('TIFF strings must be 7-bit ASCII') from exc
        elif not isinstance(value, bytes):
            raise ValueError('TIFF strings must be 7-bit ASCII')
        if len(value) == 0 or value[-1:] != b'\x00':
            value += b'\x00'
        count = len(value)
        if code == 270:
            rawcount = int(value.find(b'\x00\x00'))
            if rawcount < 0:
                rawcount = count
            else:
                rawcount = max(self.tiff.offsetsize + 1, rawcount + 1)
                rawcount = min(count, rawcount)
        else:
            rawcount = count
        value = (value,)
    elif isinstance(value, bytes):
        itemsize = struct.calcsize(dataformat)
        if len(value) % itemsize:
            raise ValueError('invalid packed binary data')
        count = len(value) // itemsize
        rawcount = count
    elif count is None:
        raise ValueError('invalid count')
    else:
        count = int(count)
    if datatype in {5, 10}:
        count *= 2
        dataformat = dataformat[-1]
    ifdentry = [pack('HH', code, datatype), pack(self.tiff.offsetformat, rawcount)]
    ifdvalue = None
    if struct.calcsize(dataformat) * count <= self.tiff.offsetsize:
        valueformat = f'{self.tiff.offsetsize}s'
        if isinstance(value, bytes):
            ifdentry.append(pack(valueformat, value))
        elif count == 1:
            if isinstance(value, (tuple, list, numpy.ndarray)):
                value = value[0]
            ifdentry.append(pack(valueformat, pack(dataformat, value)))
        else:
            ifdentry.append(pack(valueformat, pack(f'{count}{dataformat}', *value)))
    else:
        ifdentry.append(pack(self.tiff.offsetformat, 0))
        if isinstance(value, bytes):
            ifdvalue = value
        elif isinstance(value, numpy.ndarray):
            if value.size != count:
                raise RuntimeError('value.size != count')
            if value.dtype.char != dataformat:
                raise RuntimeError('value.dtype.char != dtype')
            ifdvalue = value.tobytes()
        elif isinstance(value, (tuple, list)):
            ifdvalue = pack(f'{count}{dataformat}', *value)
        else:
            ifdvalue = pack(dataformat, value)
    tags.append((code, b''.join(ifdentry), ifdvalue, writeonce))