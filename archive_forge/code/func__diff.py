from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def _diff(self, text_diff, from_label, tree, to_label, to_entry, to_tree, output_to, reverse=False):
    """See InventoryEntry._diff."""
    from breezy.diff import DiffSymlink
    old_target = self.symlink_target
    if to_entry is not None:
        new_target = to_entry.symlink_target
    else:
        new_target = None
    if not reverse:
        old_tree = tree
        new_tree = to_tree
    else:
        old_tree = to_tree
        new_tree = tree
        new_target, old_target = (old_target, new_target)
    differ = DiffSymlink(old_tree, new_tree, output_to)
    return differ.diff_symlink(old_target, new_target)