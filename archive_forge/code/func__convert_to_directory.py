import os
import re
from collections import deque
from typing import TYPE_CHECKING, Optional, Type
from .. import branch as _mod_branch
from .. import controldir, debug, errors, lazy_import, osutils, revision, trace
from .. import transport as _mod_transport
from ..controldir import ControlDir
from ..mutabletree import MutableTree
from ..repository import Repository
from ..revisiontree import RevisionTree
from breezy import (
from breezy.bzr import (
from ..tree import (FileTimestampUnavailable, InterTree, MissingNestedTree,
def _convert_to_directory(self, this_ie, inv_path):
    """Convert an entry to a directory.

        :param this_ie: Inventory entry
        :param inv_path: Normalized path for the inventory entry
        :return: The new inventory entry
        """
    this_ie = _mod_inventory.InventoryDirectory(this_ie.file_id, this_ie.name, this_ie.parent_id)
    self._invdelta[inv_path] = (inv_path, inv_path, this_ie.file_id, this_ie)
    return this_ie