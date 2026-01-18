from pickle import PicklingError
import re
import os
import os.path
import datetime
import json
import shutil
import warnings
import collections
import operator
import threading
from abc import ABCMeta, abstractmethod
from .backports import concurrency_safe_rename
from .disk import mkdirp, memstr_to_bytes, rm_subdirs
from . import numpy_pickle
def _concurrency_safe_write(self, to_write, filename, write_func):
    """Writes an object into a file in a concurrency-safe way."""
    temporary_filename = concurrency_safe_write(to_write, filename, write_func)
    self._move_item(temporary_filename, filename)