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
class AggregateIndex:
    """An aggregated index for the RepositoryPackCollection.

    AggregateIndex is reponsible for managing the PackAccess object,
    Index-To-Pack mapping, and all indices list for a specific type of index
    such as 'revision index'.

    A CombinedIndex provides an index on a single key space built up
    from several on-disk indices.  The AggregateIndex builds on this
    to provide a knit access layer, and allows having up to one writable
    index within the collection.
    """

    def __init__(self, reload_func=None, flush_func=None):
        """Create an AggregateIndex.

        :param reload_func: A function to call if we find we are missing an
            index. Should have the form reload_func() => True if the list of
            active pack files has changed.
        """
        self._reload_func = reload_func
        self.index_to_pack = {}
        self.combined_index = CombinedGraphIndex([], reload_func=reload_func)
        self.data_access = _DirectPackAccess(self.index_to_pack, reload_func=reload_func, flush_func=flush_func)
        self.add_callback = None

    def add_index(self, index, pack):
        """Add index to the aggregate, which is an index for Pack pack.

        Future searches on the aggregate index will seach this new index
        before all previously inserted indices.

        :param index: An Index for the pack.
        :param pack: A Pack instance.
        """
        self.index_to_pack[index] = pack.access_tuple()
        self.combined_index.insert_index(0, index, pack.name)

    def add_writable_index(self, index, pack):
        """Add an index which is able to have data added to it.

        There can be at most one writable index at any time.  Any
        modifications made to the knit are put into this index.

        :param index: An index from the pack parameter.
        :param pack: A Pack instance.
        """
        if self.add_callback is not None:
            raise AssertionError('%s already has a writable index through %s' % (self, self.add_callback))
        self.add_index(index, pack)
        self.data_access.set_writer(pack._writer, index, pack.access_tuple())
        self.add_callback = index.add_nodes

    def clear(self):
        """Reset all the aggregate data to nothing."""
        self.data_access.set_writer(None, None, (None, None))
        self.index_to_pack.clear()
        del self.combined_index._indices[:]
        del self.combined_index._index_names[:]
        self.add_callback = None

    def remove_index(self, index):
        """Remove index from the indices used to answer queries.

        :param index: An index from the pack parameter.
        """
        del self.index_to_pack[index]
        pos = self.combined_index._indices.index(index)
        del self.combined_index._indices[pos]
        del self.combined_index._index_names[pos]
        if self.add_callback is not None and getattr(index, 'add_nodes', None) == self.add_callback:
            self.add_callback = None
            self.data_access.set_writer(None, None, (None, None))