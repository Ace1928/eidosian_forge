import atexit
import contextlib
import functools
import inspect
import io
import os
import platform
import sys
import threading
import traceback
import debugpy
from debugpy.common import json, timestamp, util
def _update_levels():
    global _levels
    _levels = frozenset((level for file in _files.values() for level in file.levels))