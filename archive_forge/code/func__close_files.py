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
@atexit.register
def _close_files():
    for file in tuple(_files.values()):
        file.close()