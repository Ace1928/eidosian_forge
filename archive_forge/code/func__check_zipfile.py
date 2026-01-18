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
def _check_zipfile(fp):
    try:
        if _EndRecData(fp):
            return True
    except OSError:
        pass
    return False