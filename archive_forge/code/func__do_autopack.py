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
def _do_autopack(self):
    total_revisions = self.revision_index.combined_index.key_count()
    total_packs = len(self._names)
    if self._max_pack_count(total_revisions) >= total_packs:
        return None
    pack_distribution = self.pack_distribution(total_revisions)
    existing_packs = []
    for pack in self.all_packs():
        revision_count = pack.get_revision_count()
        if revision_count == 0:
            continue
        existing_packs.append((revision_count, pack))
    pack_operations = self.plan_autopack_combinations(existing_packs, pack_distribution)
    num_new_packs = len(pack_operations)
    num_old_packs = sum([len(po[1]) for po in pack_operations])
    num_revs_affected = sum([po[0] for po in pack_operations])
    mutter('Auto-packing repository %s, which has %d pack files, containing %d revisions. Packing %d files into %d affecting %d revisions', str(self), total_packs, total_revisions, num_old_packs, num_new_packs, num_revs_affected)
    result = self._execute_pack_operations(pack_operations, packer_class=self.normal_packer_class, reload_func=self._restart_autopack)
    mutter('Auto-packing repository %s completed', str(self))
    return result