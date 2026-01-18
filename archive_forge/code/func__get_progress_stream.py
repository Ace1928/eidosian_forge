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
def _get_progress_stream(self, source_vf, keys, message, pb):

    def pb_stream():
        substream = source_vf.get_record_stream(keys, 'groupcompress', True)
        for idx, record in enumerate(substream):
            if pb is not None:
                pb.update(message, idx + 1, len(keys))
            yield record
    return pb_stream()