import os
import sys
from os.path import pardir, realpath
def _safe_realpath(path):
    try:
        return realpath(path)
    except OSError:
        return path