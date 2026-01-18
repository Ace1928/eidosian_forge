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
@classmethod
def _sanitize_windows_name(cls, arcname, pathsep):
    """Replace bad characters and remove trailing dots from parts."""
    table = cls._windows_illegal_name_trans_table
    if not table:
        illegal = ':<>|"?*'
        table = str.maketrans(illegal, '_' * len(illegal))
        cls._windows_illegal_name_trans_table = table
    arcname = arcname.translate(table)
    arcname = (x.rstrip('.') for x in arcname.split(pathsep))
    arcname = pathsep.join((x for x in arcname if x))
    return arcname