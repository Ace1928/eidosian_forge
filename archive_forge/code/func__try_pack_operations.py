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
def _try_pack_operations(self, hint):
    """Calculate the pack operations based on the hint (if any), and
        execute them.
        """
    pack_operations = [[0, []]]
    for pack in self.all_packs():
        if hint is None or pack.name in hint:
            pack_operations[-1][0] += pack.get_revision_count()
            pack_operations[-1][1].append(pack)
    self._execute_pack_operations(pack_operations, packer_class=self.optimising_packer_class, reload_func=self._restart_pack_operations)