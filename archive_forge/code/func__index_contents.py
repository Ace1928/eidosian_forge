from .. import errors
from .. import transport as _mod_transport
from ..lazy_import import lazy_import
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.knit import (
from ..bzr import btree_index
from ..bzr.index import (CombinedGraphIndex, GraphIndex,
from ..bzr.vf_repository import StreamSource
from .knitrepo import KnitRepository
from .pack_repo import (NewPack, PackCommitBuilder, Packer, PackRepository,
def _index_contents(self, indices, key_filter=None):
    """Get an iterable of the index contents from a pack_map.

        :param indices: The list of indices to query
        :param key_filter: An optional filter to limit the keys returned.
        """
    all_index = CombinedGraphIndex(indices)
    if key_filter is None:
        return all_index.iter_all_entries()
    else:
        return all_index.iter_entries(key_filter)