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
def _exhaust_stream(self, source_vf, keys, message, vf_to_stream, pb_offset):
    """Create and exhaust a stream, but don't insert it.

        This is useful to get the side-effects of generating a stream.
        """
    self.pb.update('scanning {}'.format(message), pb_offset)
    with ui.ui_factory.nested_progress_bar() as child_pb:
        list(vf_to_stream(source_vf, keys, message, child_pb))