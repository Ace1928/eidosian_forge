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
def _fix_case_of_inventory_path(self, path):
    """If our tree isn't case sensitive, return the canonical path"""
    if not self.case_sensitive:
        path = self.get_canonical_path(path)
    return path