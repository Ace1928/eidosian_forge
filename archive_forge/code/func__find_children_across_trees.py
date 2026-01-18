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
def _find_children_across_trees(specified_ids, trees):
    """Return a set including specified ids and their children.

    All matches in all trees will be used.

    :param trees: The trees to find file_ids within
    :return: a set containing all specified ids and their children
    """
    interesting_ids = set(specified_ids)
    pending = interesting_ids
    while len(pending) > 0:
        new_pending = set()
        for file_id in pending:
            for tree in trees:
                try:
                    path = tree.id2path(file_id)
                except errors.NoSuchId:
                    continue
                try:
                    for child in tree.iter_child_entries(path):
                        if child.file_id not in interesting_ids:
                            new_pending.add(child.file_id)
                except errors.NotADirectory:
                    pass
        interesting_ids.update(new_pending)
        pending = new_pending
    return interesting_ids