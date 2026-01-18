from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
class Decompressors(object):
    """Delay import decompressor functions."""

    def __init__(self):
        self._decompressors = {None: identityfunc, 1: identityfunc, 5: decode_lzw, 8: zlib.decompress, 32773: decode_packbits, 32946: zlib.decompress}

    def __getitem__(self, key):
        if key in self._decompressors:
            return self._decompressors[key]
        if key == 7:
            try:
                from imagecodecs import jpeg, jpeg_12
            except ImportError:
                raise KeyError

            def decode_jpeg(x, table, bps, colorspace=None):
                if bps == 8:
                    return jpeg.decode_jpeg(x, table, colorspace)
                elif bps == 12:
                    return jpeg_12.decode_jpeg_12(x, table, colorspace)
                else:
                    raise ValueError('bitspersample not supported')
            self._decompressors[key] = decode_jpeg
            return decode_jpeg
        if key == 34925:
            try:
                import lzma
            except ImportError:
                try:
                    import backports.lzma as lzma
                except ImportError:
                    raise KeyError
            self._decompressors[key] = lzma.decompress
            return lzma.decompress
        if key == 34926:
            try:
                import zstd
            except ImportError:
                raise KeyError
            self._decompressors[key] = zstd.decompress
            return zstd.decompress
        raise KeyError

    def __contains__(self, item):
        try:
            self[item]
            return True
        except KeyError:
            return False