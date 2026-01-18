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
def _init_decrypter(self):
    self._decrypter = _ZipDecrypter(self._pwd)
    header = self._fileobj.read(12)
    self._compress_left -= 12
    return self._decrypter(header)[11]