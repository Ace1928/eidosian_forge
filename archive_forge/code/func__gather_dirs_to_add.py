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
def _gather_dirs_to_add(self, user_dirs):
    prev_dir = None
    is_inside = osutils.is_inside_or_parent_of_any
    for path in sorted(user_dirs):
        if prev_dir is None or not is_inside([prev_dir], path):
            inv_path, this_ie = user_dirs[path]
            yield (path, inv_path, this_ie, None)
        prev_dir = path