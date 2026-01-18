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
def __zbox(self, s):
    z = []
    for p in s.points:
        try:
            z.append(p[2])
        except IndexError:
            z.append(0)
    zbox = [min(z), max(z)]
    if self._zbox:
        self._zbox = [min(zbox[0], self._zbox[0]), max(zbox[1], self._zbox[1])]
    else:
        self._zbox = zbox
    return zbox