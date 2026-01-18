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
def _writable_required(self, path):
    """Check that ``path`` is writeable."""
    if self.write_fs is None:
        raise errors.ResourceReadOnly(path)
    return self.write_fs