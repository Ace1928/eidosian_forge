from __future__ import absolute_import, print_function, unicode_literals
import sys
import typing
import errno
import io
import itertools
import logging
import os
import platform
import shutil
import six
import stat
import tempfile
from . import errors
from ._fscompat import fsdecode, fsencode, fspath
from ._url_tools import url_quote
from .base import FS
from .copy import copy_modified_time
from .enums import ResourceType
from .error_tools import convert_os_errors
from .errors import FileExpected, NoURL
from .info import Info
from .mode import Mode, validate_open_mode
from .path import basename, dirname
from .permissions import Permissions
@classmethod
def _make_access_from_stat(cls, stat_result):
    """Make an *access* info dict from an `os.stat_result` object."""
    access = {}
    access['permissions'] = Permissions(mode=stat_result.st_mode).dump()
    access['gid'] = gid = stat_result.st_gid
    access['uid'] = uid = stat_result.st_uid
    if not _WINDOWS_PLATFORM:
        import grp
        import pwd
        try:
            access['group'] = grp.getgrgid(gid).gr_name
        except KeyError:
            pass
        try:
            access['user'] = pwd.getpwuid(uid).pw_name
        except KeyError:
            pass
    return access