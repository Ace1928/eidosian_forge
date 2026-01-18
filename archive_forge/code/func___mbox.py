from struct import pack, unpack, calcsize, error, Struct
import os
import sys
import time
import array
import tempfile
import logging
import io
from datetime import date
import zipfile
def __mbox(self, s):
    mpos = 3 if s.shapeType in (11, 13, 15, 18, 31) else 2
    m = []
    for p in s.points:
        try:
            if p[mpos] is not None:
                m.append(p[mpos])
        except IndexError:
            pass
    if not m:
        m.append(NODATA)
    mbox = [min(m), max(m)]
    if self._mbox:
        self._mbox = [min(mbox[0], self._mbox[0]), max(mbox[1], self._mbox[1])]
    else:
        self._mbox = mbox
    return mbox