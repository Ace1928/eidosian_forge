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
def _make_index(self, name, suffix, resume=False, is_chk=False):
    size_offset = self._suffix_offsets[suffix]
    index_name = name + suffix
    if resume:
        transport = self._upload_transport
        index_size = transport.stat(index_name).st_size
    else:
        transport = self._index_transport
        index_size = self._names[name][size_offset]
    index = self._index_class(transport, index_name, index_size, unlimited_cache=is_chk)
    if is_chk and self._index_class is btree_index.BTreeGraphIndex:
        index._leaf_factory = btree_index._gcchk_factory
    return index