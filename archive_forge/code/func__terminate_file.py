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
def _terminate_file(self, *, is_rotating=False):
    old_path = self._file_path
    if self._file is not None:
        self._close_file()
    if is_rotating:
        new_path = self._create_path()
        self._create_dirs(new_path)
        if new_path == old_path:
            creation_time = get_ctime(old_path)
            root, ext = os.path.splitext(old_path)
            renamed_path = generate_rename_path(root, ext, creation_time)
            os.rename(old_path, renamed_path)
            old_path = renamed_path
    if is_rotating or self._rotation_function is None:
        if self._compression_function is not None and old_path is not None:
            self._compression_function(old_path)
        if self._retention_function is not None:
            logs = {file for pattern in self._glob_patterns for file in glob.glob(pattern) if os.path.isfile(file)}
            self._retention_function(list(logs))
    if is_rotating:
        self._create_file(new_path)
        set_ctime(new_path, datetime.datetime.now().timestamp())