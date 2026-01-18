from __future__ import print_function, unicode_literals
import typing
import warnings
from .errors import ResourceNotFound
from .opener import manage_fs
from .path import abspath, combine, frombase, normpath
from .tools import is_thread_safe
from .walk import Walker
def _copy_locked():
    if dst_fs.hassyspath(dst_path):
        with dst_fs.openbin(dst_path, 'w') as write_file:
            src_fs.download(src_path, write_file)
    else:
        with src_fs.openbin(src_path) as read_file:
            dst_fs.upload(dst_path, read_file)
    if preserve_time:
        copy_modified_time(src_fs, src_path, dst_fs, dst_path)