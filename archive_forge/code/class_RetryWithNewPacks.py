import re
import sys
from typing import Type
from ..lazy_import import lazy_import
import contextlib
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.index import (
from .. import errors, lockable_files, lockdir
from .. import transport as _mod_transport
from ..bzr import btree_index, index
from ..decorators import only_raises
from ..lock import LogicalLockResult
from ..repository import RepositoryWriteLockResult, _LazyListJoin
from ..trace import mutter, note, warning
from .repository import MetaDirRepository, RepositoryFormatMetaDir
from .serializer import Serializer
from .vf_repository import (MetaDirVersionedFileRepository,
class RetryWithNewPacks(errors.BzrError):
    """Raised when we realize that the packs on disk have changed.

    This is meant as more of a signaling exception, to trap between where a
    local error occurred and the code that can actually handle the error and
    code that can retry appropriately.
    """
    internal_error = True
    _fmt = 'Pack files have changed, reload and retry. context: %(context)s %(orig_error)s'

    def __init__(self, context, reload_occurred, exc_info):
        """create a new RetryWithNewPacks error.

        :param reload_occurred: Set to True if we know that the packs have
            already been reloaded, and we are failing because of an in-memory
            cache miss. If set to True then we will ignore if a reload says
            nothing has changed, because we assume it has already reloaded. If
            False, then a reload with nothing changed will force an error.
        :param exc_info: The original exception traceback, so if there is a
            problem we can raise the original error (value from sys.exc_info())
        """
        errors.BzrError.__init__(self)
        self.context = context
        self.reload_occurred = reload_occurred
        self.exc_info = exc_info
        self.orig_error = exc_info[1]