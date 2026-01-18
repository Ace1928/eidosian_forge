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
class _InterestingKeyInfo:

    def __init__(self):
        self.interesting_root_keys = set()
        self.interesting_pid_root_keys = set()
        self.uninteresting_root_keys = set()
        self.uninteresting_pid_root_keys = set()

    def all_interesting(self):
        return self.interesting_root_keys.union(self.interesting_pid_root_keys)

    def all_uninteresting(self):
        return self.uninteresting_root_keys.union(self.uninteresting_pid_root_keys)

    def all_keys(self):
        return self.all_interesting().union(self.all_uninteresting())