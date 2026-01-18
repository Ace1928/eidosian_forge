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
def _obsolete_packs(self, packs):
    """Move a number of packs which have been obsoleted out of the way.

        Each pack and its associated indices are moved out of the way.

        Note: for correctness this function should only be called after a new
        pack names index has been written without these pack names, and with
        the names of packs that contain the data previously available via these
        packs.

        :param packs: The packs to obsolete.
        :param return: None.
        """
    for pack in packs:
        try:
            try:
                pack.pack_transport.move(pack.file_name(), '../obsolete_packs/' + pack.file_name())
            except _mod_transport.NoSuchFile:
                try:
                    pack.pack_transport.mkdir('../obsolete_packs/')
                except _mod_transport.FileExists:
                    pass
                pack.pack_transport.move(pack.file_name(), '../obsolete_packs/' + pack.file_name())
        except (errors.PathError, errors.TransportError) as e:
            mutter("couldn't rename obsolete pack, skipping it:\n%s" % (e,))
        suffixes = ['.iix', '.six', '.tix', '.rix']
        if self.chk_index is not None:
            suffixes.append('.cix')
        for suffix in suffixes:
            try:
                self._index_transport.move(pack.name + suffix, '../obsolete_packs/' + pack.name + suffix)
            except (errors.PathError, errors.TransportError) as e:
                mutter("couldn't rename obsolete index, skipping it:\n%s" % (e,))