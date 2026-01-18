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
def COMPESSORS():

    class Compressors(object):
        """Delay import compressor functions."""

        def __init__(self):
            self._compressors = {8: (zlib.compress, 6), 32946: (zlib.compress, 6)}

        def __getitem__(self, key):
            if key in self._compressors:
                return self._compressors[key]
            if key == 34925:
                try:
                    import lzma
                except ImportError:
                    try:
                        import backports.lzma as lzma
                    except ImportError:
                        raise KeyError

                def lzma_compress(x, level):
                    return lzma.compress(x)
                self._compressors[key] = (lzma_compress, 0)
                return (lzma_compress, 0)
            if key == 34926:
                try:
                    import zstd
                except ImportError:
                    raise KeyError
                self._compressors[key] = (zstd.compress, 9)
                return (zstd.compress, 9)
            raise KeyError

        def __contains__(self, key):
            try:
                self[key]
                return True
            except KeyError:
                return False
    return Compressors()