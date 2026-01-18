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
def _encodeFilenameFlags(self):
    try:
        return (self.filename.encode('ascii'), self.flag_bits)
    except UnicodeEncodeError:
        return (self.filename.encode('utf-8'), self.flag_bits | _MASK_UTF_FILENAME)