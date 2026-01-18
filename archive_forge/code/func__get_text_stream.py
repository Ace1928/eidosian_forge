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
def _get_text_stream(self):
    text_stream = self.from_repository.texts.get_record_stream(self._text_keys, self._text_fetch_order, False)
    return ('texts', text_stream)