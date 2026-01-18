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
def _max_pack_count(self, total_revisions):
    """Return the maximum number of packs to use for total revisions.

        :param total_revisions: The total number of revisions in the
            repository.
        """
    if not total_revisions:
        return 1
    digits = str(total_revisions)
    result = 0
    for digit in digits:
        result += int(digit)
    return result