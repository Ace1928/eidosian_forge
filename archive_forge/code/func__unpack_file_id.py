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
def _unpack_file_id(self, file_id):
    """Find the inventory and inventory file id for a tree file id.

        :param file_id: The tree file id, as bytestring or tuple
        :return: Inventory and inventory file id
        """
    if isinstance(file_id, tuple):
        if len(file_id) != 1:
            raise ValueError('nested trees not yet supported: %r' % file_id)
        file_id = file_id[0]
    return (self.root_inventory, file_id)