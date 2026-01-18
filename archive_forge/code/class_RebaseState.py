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
class RebaseState:

    def has_plan(self):
        """Check whether there is a rebase plan present.

        :return: boolean
        """
        raise NotImplementedError(self.has_plan)

    def read_plan(self):
        """Read a rebase plan file.

        :return: Tuple with last revision info and replace map.
        """
        raise NotImplementedError(self.read_plan)

    def write_plan(self, replace_map):
        """Write a rebase plan file.

        :param replace_map: Replace map (old revid -> (new revid, new parents))
        """
        raise NotImplementedError(self.write_plan)

    def remove_plan(self):
        """Remove a rebase plan file.
        """
        raise NotImplementedError(self.remove_plan)

    def write_active_revid(self, revid):
        """Write the id of the revision that is currently being rebased.

        :param revid: Revision id to write
        """
        raise NotImplementedError(self.write_active_revid)

    def read_active_revid(self):
        """Read the id of the revision that is currently being rebased.

        :return: Id of the revision that is being rebased.
        """
        raise NotImplementedError(self.read_active_revid)