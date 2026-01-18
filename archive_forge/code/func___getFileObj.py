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
def __getFileObj(self, f):
    """Safety handler to verify file-like objects"""
    if not f:
        raise ShapefileException('No file-like object available.')
    elif hasattr(f, 'write'):
        return f
    else:
        pth = os.path.split(f)[0]
        if pth and (not os.path.exists(pth)):
            os.makedirs(pth)
        fp = open(f, 'wb+')
        self._files_to_close.append(fp)
        return fp