import datetime
import decimal
import glob
import numbers
import os
import shutil
import string
from functools import partial
from stat import ST_DEV, ST_INO
from . import _string_parsers as string_parsers
from ._ctime_functions import get_ctime, set_ctime
from ._datetime import aware_now
def _reopen_if_needed(self):
    if not self._file:
        return
    filepath = self._file_path
    try:
        result = os.stat(filepath)
    except FileNotFoundError:
        result = None
    if not result or result[ST_DEV] != self._file_dev or result[ST_INO] != self._file_ino:
        self._close_file()
        self._create_dirs(filepath)
        self._create_file(filepath)