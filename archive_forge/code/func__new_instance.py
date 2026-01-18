import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo
from . import _common, _tzpath
@classmethod
def _new_instance(cls, key):
    obj = super().__new__(cls)
    obj._key = key
    obj._file_path = obj._find_tzfile(key)
    if obj._file_path is not None:
        file_obj = open(obj._file_path, 'rb')
    else:
        file_obj = _common.load_tzdata(key)
    with file_obj as f:
        obj._load_file(f)
    return obj