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
def _filtered_inv_stream():
    source_vf = from_repo.inventories
    stream = source_vf.get_record_stream(revision_keys, 'unordered', False)
    for record in stream:
        if record.storage_kind == 'absent':
            raise errors.NoSuchRevision(from_repo, record.key)
        find_text_keys_from_content(record)
        yield record
    self._text_keys = content_text_keys - parent_text_keys