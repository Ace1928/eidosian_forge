import glob
import fnmatch
import string
import json
import os
import os.path as op
import shutil
import subprocess
import re
import copy
import tempfile
from os.path import join, dirname
from warnings import warn
from .. import config, logging
from ..utils.filemanip import (
from ..utils.misc import human_order_sorted, str2bool
from .base import (
class ProgressPercentage(object):
    """
    Callable class instsance (via __call__ method) that displays
    upload percentage of a file to S3
    """

    def __init__(self, filename):
        """ """
        import threading
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        """ """
        import sys
        with self._lock:
            self._seen_so_far += bytes_amount
            if self._size != 0:
                percentage = self._seen_so_far // self._size * 100
            else:
                percentage = 0
            progress_str = '%d / %d (%.2f%%)\r' % (self._seen_so_far, self._size, percentage)
            sys.stdout.write(progress_str)
            sys.stdout.flush()