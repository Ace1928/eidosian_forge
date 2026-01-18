from __future__ import annotations
import sys
from collections.abc import Iterator, Mapping
from pathlib import PurePosixPath
from typing import (
from xarray.core.utils import Frozen, is_dict_like
class NamedNode(TreeNode, Generic[Tree]):
    """
    A TreeNode which knows its own name.

    Implements path-like relationships to other nodes in its tree.
    """
    _name: str | None
    _parent: Tree | None
    _children: dict[str, Tree]

    def __init__(self, name=None, children=None):
        super().__init__(children=children)
        self._name = None
        self.name = name

    @property
    def name(self) -> str | None:
        """The name of this node."""
        return self._name

    @name.setter
    def name(self, name: str | None) -> None:
        if name is not None:
            if not isinstance(name, str):
                raise TypeError('node name must be a string or None')
            if '/' in name:
                raise ValueError('node names cannot contain forward slashes')
        self._name = name

    def __repr__(self, level=0):
        repr_value = '\t' * level + self.__str__() + '\n'
        for child in self.children:
            repr_value += self.get(child).__repr__(level + 1)
        return repr_value

    def __str__(self) -> str:
        return f"NamedNode('{self.name}')" if self.name else 'NamedNode()'

    def _post_attach(self: NamedNode, parent: NamedNode) -> None:
        """Ensures child has name attribute corresponding to key under which it has been stored."""
        key = next((k for k, v in parent.children.items() if v is self))
        self.name = key

    @property
    def path(self) -> str:
        """Return the file-like path from the root to this node."""
        if self.is_root:
            return '/'
        else:
            root, *ancestors = tuple(reversed(self.parents))
            names = [*(node.name for node in ancestors), self.name]
            return '/' + '/'.join(names)

    def relative_to(self: NamedNode, other: NamedNode) -> str:
        """
        Compute the relative path from this node to node `other`.

        If other is not in this tree, or it's otherwise impossible, raise a ValueError.
        """
        if not self.same_tree(other):
            raise NotFoundInTreeError('Cannot find relative path because nodes do not lie within the same tree')
        this_path = NodePath(self.path)
        if other.path in list((parent.path for parent in (self, *self.parents))):
            return str(this_path.relative_to(other.path))
        else:
            common_ancestor = self.find_common_ancestor(other)
            path_to_common_ancestor = other._path_to_ancestor(common_ancestor)
            return str(path_to_common_ancestor / this_path.relative_to(common_ancestor.path))

    def find_common_ancestor(self, other: NamedNode) -> NamedNode:
        """
        Find the first common ancestor of two nodes in the same tree.

        Raise ValueError if they are not in the same tree.
        """
        if self is other:
            return self
        other_paths = [op.path for op in other.parents]
        for parent in (self, *self.parents):
            if parent.path in other_paths:
                return parent
        raise NotFoundInTreeError('Cannot find common ancestor because nodes do not lie within the same tree')

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