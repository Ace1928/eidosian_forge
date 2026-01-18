from __future__ import annotations
import sys
from collections.abc import Iterator, Mapping
from pathlib import PurePosixPath
from typing import (
from xarray.core.utils import Frozen, is_dict_like
def _path_to_ancestor(self, ancestor: NamedNode) -> NodePath:
    """Return the relative path from this node to the given ancestor node"""
    if not self.same_tree(ancestor):
        raise NotFoundInTreeError('Cannot find relative path to ancestor because nodes do not lie within the same tree')
    if ancestor.path not in list((a.path for a in (self, *self.parents))):
        raise NotFoundInTreeError('Cannot find relative path to ancestor because given node is not an ancestor of this node')
    parents_paths = list((parent.path for parent in (self, *self.parents)))
    generation_gap = list(parents_paths).index(ancestor.path)
    path_upwards = '../' * generation_gap if generation_gap > 0 else '.'
    return NodePath(path_upwards)