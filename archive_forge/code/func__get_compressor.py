import binascii
import importlib.util
import io
import itertools
import os
import posixpath
import shutil
import stat
import struct
import sys
import threading
import time
import contextlib
import pathlib
def _get_compressor(compress_type, compresslevel=None):
    if compress_type == ZIP_DEFLATED:
        if compresslevel is not None:
            return zlib.compressobj(compresslevel, zlib.DEFLATED, -15)
        return zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -15)
    elif compress_type == ZIP_BZIP2:
        if compresslevel is not None:
            return bz2.BZ2Compressor(compresslevel)
        return bz2.BZ2Compressor()
    elif compress_type == ZIP_LZMA:
        return LZMACompressor()
    else:
        return None