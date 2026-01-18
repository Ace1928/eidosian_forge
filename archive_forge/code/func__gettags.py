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
def _gettags(self, codes: Container[int] | None=None, /, lock: threading.RLock | None=None) -> list[tuple[int, TiffTag]]:
    """Return list of (code, TiffTag) from file."""
    fh = self.parent.filehandle
    tiff = self.parent.tiff
    unpack = struct.unpack
    rlock: Any = NullContext() if lock is None else lock
    tags = []
    with rlock:
        fh.seek(self.offset)
        try:
            tagno = unpack(tiff.tagnoformat, fh.read(tiff.tagnosize))[0]
            if tagno > 4096:
                raise ValueError(f'suspicious number of tags {tagno}')
        except Exception as exc:
            raise TiffFileError(f'corrupted tag list @{self.offset}') from exc
        tagoffset = self.offset + tiff.tagnosize
        tagsize = tiff.tagsize
        tagindex = -tagsize
        codeformat = tiff.tagformat1[:2]
        tagbytes = fh.read(tagsize * tagno)
        for _ in range(tagno):
            tagindex += tagsize
            code = unpack(codeformat, tagbytes[tagindex:tagindex + 2])[0]
            if codes and code not in codes:
                continue
            try:
                tag = TiffTag.fromfile(self.parent, offset=tagoffset + tagindex, header=tagbytes[tagindex:tagindex + tagsize])
            except TiffFileError as exc:
                logger().error(f'{self!r} <TiffTag.fromfile> raised {exc!r}')
                continue
            tags.append((code, tag))
    return tags