import time
from .. import controldir, debug, errors, osutils
from .. import revision as _mod_revision
from .. import trace, ui
from ..bzr import chk_map, chk_serializer
from ..bzr import index as _mod_index
from ..bzr import inventory, pack, versionedfile
from ..bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from ..bzr.groupcompress import GroupCompressVersionedFiles, _GCGraphIndex
from ..bzr.vf_repository import StreamSource
from .pack_repo import (NewPack, Pack, PackCommitBuilder, Packer,
from .static_tuple import StaticTuple
def _build_vfs(self, index_name, parents, delta):
    """Build the source and target VersionedFiles."""
    source_vf = self._build_vf(index_name, parents, delta, for_write=False)
    target_vf = self._build_vf(index_name, parents, delta, for_write=True)
    return (source_vf, target_vf)