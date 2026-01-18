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
def _extract_text_encoding(encoding=None, *args, **kwargs):
    return (io.text_encoding(encoding, 3), args, kwargs)