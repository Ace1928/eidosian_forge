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
def _pack_map_and_index_list(self, index_attribute):
    """Convert a list of packs to an index pack map and index list.

        :param index_attribute: The attribute that the desired index is found
            on.
        :return: A tuple (map, list) where map contains the dict from
            index:pack_tuple, and list contains the indices in the preferred
            access order.
        """
    indices = []
    pack_map = {}
    for pack_obj in self.packs:
        index = getattr(pack_obj, index_attribute)
        indices.append(index)
        pack_map[index] = pack_obj
    return (pack_map, indices)