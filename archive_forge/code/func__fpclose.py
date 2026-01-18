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
def _fpclose(self, fp):
    assert self._fileRefCnt > 0
    self._fileRefCnt -= 1
    if not self._fileRefCnt and (not self._filePassed):
        fp.close()