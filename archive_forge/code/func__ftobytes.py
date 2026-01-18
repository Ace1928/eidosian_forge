import errno
import glob
import hashlib
import importlib.metadata as importlib_metadata
import itertools
import json
import logging
import os
import os.path
import struct
import sys
def _ftobytes(f):
    return struct.Struct('f').pack(f)