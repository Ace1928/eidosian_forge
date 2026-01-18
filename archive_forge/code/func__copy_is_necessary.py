from __future__ import print_function, unicode_literals
import typing
import warnings
from .errors import ResourceNotFound
from .opener import manage_fs
from .path import abspath, combine, frombase, normpath
from .tools import is_thread_safe
from .walk import Walker
def _copy_is_necessary(src_fs, src_path, dst_fs, dst_path, condition):
    if condition == 'always':
        return True
    elif condition == 'newer':
        try:
            src_modified = src_fs.getmodified(src_path)
            dst_modified = dst_fs.getmodified(dst_path)
        except ResourceNotFound:
            return True
        else:
            return src_modified is None or dst_modified is None or src_modified > dst_modified
    elif condition == 'older':
        try:
            src_modified = src_fs.getmodified(src_path)
            dst_modified = dst_fs.getmodified(dst_path)
        except ResourceNotFound:
            return True
        else:
            return src_modified is None or dst_modified is None or src_modified < dst_modified
    elif condition == 'exists':
        return dst_fs.exists(dst_path)
    elif condition == 'not_exists':
        return not dst_fs.exists(dst_path)
    else:
        raise ValueError('{} is not a valid copy condition.'.format(condition))