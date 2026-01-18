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
def _base(self):
    return pathlib.PurePosixPath(self.at or self.root.filename)