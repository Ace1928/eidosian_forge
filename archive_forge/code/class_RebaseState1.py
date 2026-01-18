import os
from ... import config as _mod_config
from ... import osutils, ui
from ...bzr.generate_ids import gen_revision_id
from ...bzr.inventorytree import InventoryTreeChange
from ...errors import (BzrError, NoCommonAncestor, UnknownFormatError,
from ...graph import FrozenHeadsCache
from ...merge import Merger
from ...revision import NULL_REVISION
from ...trace import mutter
from ...transport import NoSuchFile
from ...tsort import topo_sort
from .maptree import MapTree, map_file_ids
class RebaseState1(RebaseState):

    def __init__(self, wt):
        self.wt = wt
        self.transport = wt._transport

    def has_plan(self):
        """See `RebaseState`."""
        try:
            return self.transport.get_bytes(REBASE_PLAN_FILENAME) != b''
        except NoSuchFile:
            return False

    def read_plan(self):
        """See `RebaseState`."""
        text = self.transport.get_bytes(REBASE_PLAN_FILENAME)
        if text == b'':
            raise NoSuchFile(REBASE_PLAN_FILENAME)
        return unmarshall_rebase_plan(text)

    def write_plan(self, replace_map):
        """See `RebaseState`."""
        self.wt.update_feature_flags({b'rebase-v1': b'write-required'})
        content = marshall_rebase_plan(self.wt.branch.last_revision_info(), replace_map)
        assert isinstance(content, bytes)
        self.transport.put_bytes(REBASE_PLAN_FILENAME, content)

    def remove_plan(self):
        """See `RebaseState`."""
        self.wt.update_feature_flags({b'rebase-v1': None})
        self.transport.put_bytes(REBASE_PLAN_FILENAME, b'')

    def write_active_revid(self, revid):
        """See `RebaseState`."""
        if revid is None:
            revid = NULL_REVISION
        assert isinstance(revid, bytes)
        self.transport.put_bytes(REBASE_CURRENT_REVID_FILENAME, revid)

    def read_active_revid(self):
        """See `RebaseState`."""
        try:
            text = self.transport.get_bytes(REBASE_CURRENT_REVID_FILENAME).rstrip(b'\n')
            if text == NULL_REVISION:
                return None
            return text
        except NoSuchFile:
            return None