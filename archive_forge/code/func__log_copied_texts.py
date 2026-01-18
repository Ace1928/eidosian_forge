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
def _log_copied_texts(self):
    if 'pack' in debug.debug_flags:
        mutter('%s: create_pack: file texts copied: %s%s %d items t+%6.3fs', time.ctime(), self._pack_collection._upload_transport.base, self.new_pack.random_name, self.new_pack.text_index.key_count(), time.time() - self.new_pack.start_time)