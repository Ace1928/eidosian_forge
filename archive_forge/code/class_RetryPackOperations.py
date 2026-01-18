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
class RetryPackOperations(RetryWithNewPacks):
    """Raised when we are packing and we find a missing file.

    Meant as a signaling exception, to tell the RepositoryPackCollection.pack
    code it should try again.
    """
    internal_error = True
    _fmt = 'Pack files have changed, reload and try pack again. context: %(context)s %(orig_error)s'