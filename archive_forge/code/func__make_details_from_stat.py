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
def _make_details_from_stat(cls, stat_result):
    """Make a *details* info dict from an `os.stat_result` object."""
    details = {'_write': ['accessed', 'modified'], 'accessed': stat_result.st_atime, 'modified': stat_result.st_mtime, 'size': stat_result.st_size, 'type': int(cls._get_type_from_stat(stat_result))}
    details['created'] = getattr(stat_result, 'st_birthtime', None)
    ctime_key = 'created' if _WINDOWS_PLATFORM else 'metadata_changed'
    details[ctime_key] = stat_result.st_ctime
    return details