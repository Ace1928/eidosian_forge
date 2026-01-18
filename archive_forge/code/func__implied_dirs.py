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
@staticmethod
def _implied_dirs(names):
    parents = itertools.chain.from_iterable(map(_parents, names))
    as_dirs = (p + posixpath.sep for p in parents)
    return _dedupe(_difference(as_dirs, names))