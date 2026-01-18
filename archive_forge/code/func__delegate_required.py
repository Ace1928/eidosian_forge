from __future__ import absolute_import, print_function, unicode_literals
import typing
from collections import OrderedDict, namedtuple
from operator import itemgetter
from six import text_type
from . import errors
from .base import FS
from .mode import check_writable
from .opener import open_fs
from .path import abspath, normpath
def _delegate_required(self, path):
    """Check that there is a filesystem with the given ``path``."""
    fs = self._delegate(path)
    if fs is None:
        raise errors.ResourceNotFound(path)
    return fs