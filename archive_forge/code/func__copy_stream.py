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
def _copy_stream(self, source_vf, target_vf, keys, message, vf_to_stream, pb_offset):
    trace.mutter('repacking %d %s', len(keys), message)
    self.pb.update('repacking {}'.format(message), pb_offset)
    with ui.ui_factory.nested_progress_bar() as child_pb:
        stream = vf_to_stream(source_vf, keys, message, child_pb)
        for _, _ in target_vf._insert_record_stream(stream, random_id=True, reuse_blocks=False):
            pass