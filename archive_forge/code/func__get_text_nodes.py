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
def _get_text_nodes(self):
    text_index_map, text_indices = self._pack_map_and_index_list('text_index')
    return (text_index_map, self._index_contents(text_indices, self._text_filter))