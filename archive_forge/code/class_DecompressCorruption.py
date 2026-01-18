import time
import zlib
from typing import Type
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import errors, osutils, trace
from ..lru_cache import LRUSizeCache
from .btree_index import BTreeBuilder
from .versionedfile import (AbsentContentFactory, ChunkedContentFactory,
from ._groupcompress_py import (LinesDeltaIndex, apply_delta,
class DecompressCorruption(errors.BzrError):
    _fmt = 'Corruption while decompressing repository file%(orig_error)s'

    def __init__(self, orig_error=None):
        if orig_error is not None:
            self.orig_error = ', {}'.format(orig_error)
        else:
            self.orig_error = ''
        errors.BzrError.__init__(self)