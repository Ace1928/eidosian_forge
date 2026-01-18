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
def _process_inventory_lines(self, inv_lines):
    """Generate a text key reference map rather for reconciling with."""
    repo = self._pack_collection.repo
    refs = repo._serializer._find_text_key_references(inv_lines)
    self._text_refs = refs
    self._text_filter = None