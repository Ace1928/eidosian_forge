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
def _make_link_info(self, sys_path):
    _target = self._gettarget(sys_path)
    return {'target': _target}