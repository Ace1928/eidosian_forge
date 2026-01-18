import errno
import fnmatch
import marshal
import os
import pickle
import stat
import sys
import tempfile
import typing as t
from hashlib import sha1
from io import BytesIO
from types import CodeType
def _get_default_cache_dir(self) -> str:

    def _unsafe_dir() -> 'te.NoReturn':
        raise RuntimeError('Cannot determine safe temp directory.  You need to explicitly provide one.')
    tmpdir = tempfile.gettempdir()
    if os.name == 'nt':
        return tmpdir
    if not hasattr(os, 'getuid'):
        _unsafe_dir()
    dirname = f'_jinja2-cache-{os.getuid()}'
    actual_dir = os.path.join(tmpdir, dirname)
    try:
        os.mkdir(actual_dir, stat.S_IRWXU)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    try:
        os.chmod(actual_dir, stat.S_IRWXU)
        actual_dir_stat = os.lstat(actual_dir)
        if actual_dir_stat.st_uid != os.getuid() or not stat.S_ISDIR(actual_dir_stat.st_mode) or stat.S_IMODE(actual_dir_stat.st_mode) != stat.S_IRWXU:
            _unsafe_dir()
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    actual_dir_stat = os.lstat(actual_dir)
    if actual_dir_stat.st_uid != os.getuid() or not stat.S_ISDIR(actual_dir_stat.st_mode) or stat.S_IMODE(actual_dir_stat.st_mode) != stat.S_IRWXU:
        _unsafe_dir()
    return actual_dir