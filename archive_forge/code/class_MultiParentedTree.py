import warnings
from abc import ABCMeta, abstractmethod
from nltk.tree.tree import Tree
from nltk.util import slice_bounds
class MultiParentedTree(AbstractParentedTree):
    """
    A ``Tree`` that automatically maintains parent pointers for
    multi-parented trees.  The following are methods for querying the
    structure of a multi-parented tree: ``parents()``, ``parent_indices()``,
    ``left_siblings()``, ``right_siblings()``, ``roots``, ``treepositions``.

    Each ``MultiParentedTree`` may have zero or more parents.  In
    particular, subtrees may be shared.  If a single
    ``MultiParentedTree`` is used as multiple children of the same
    parent, then that parent will appear multiple times in its
    ``parents()`` method.

    ``MultiParentedTrees`` should never be used in the same tree as
    ``Trees`` or ``ParentedTrees``.  Mixing tree implementations may
    result in incorrect parent pointers and in ``TypeError`` exceptions.
    """

    def __init__(self, node, children=None):
        self._parents = []
        "A list of this tree's parents.  This list should not\n           contain duplicates, even if a parent contains this tree\n           multiple times."
        super().__init__(node, children)
        if children is None:
            for i, child in enumerate(self):
                if isinstance(child, Tree):
                    child._parents = []
                    self._setparent(child, i)

    def _frozen_class(self):
        from nltk.tree.immutable import ImmutableMultiParentedTree
        return ImmutableMultiParentedTree

    def parents(self):
        """
        The set of parents of this tree.  If this tree has no parents,
        then ``parents`` is the empty set.  To check if a tree is used
        as multiple children of the same parent, use the
        ``parent_indices()`` method.

        :type: list(MultiParentedTree)
        """
        return list(self._parents)

    def left_siblings(self):
        """
        A list of all left siblings of this tree, in any of its parent
        trees.  A tree may be its own left sibling if it is used as
        multiple contiguous children of the same parent.  A tree may
        appear multiple times in this list if it is the left sibling
        of this tree with respect to multiple parents.

        :type: list(MultiParentedTree)
        """
        return [parent[index - 1] for parent, index in self._get_parent_indices() if index > 0]

    def right_siblings(self):
        """
        A list of all right siblings of this tree, in any of its parent
        trees.  A tree may be its own right sibling if it is used as
        multiple contiguous children of the same parent.  A tree may
        appear multiple times in this list if it is the right sibling
        of this tree with respect to multiple parents.

        :type: list(MultiParentedTree)
        """
        return [parent[index + 1] for parent, index in self._get_parent_indices() if index < len(parent) - 1]

    def _get_parent_indices(self):
        return [(parent, index) for parent in self._parents for index, child in enumerate(parent) if child is self]

    def roots(self):
        """
        The set of all roots of this tree.  This set is formed by
        tracing all possible parent paths until trees with no parents
        are found.

        :type: list(MultiParentedTree)
        """
        return list(self._get_roots_helper({}).values())

    def _get_roots_helper(self, result):
        if self._parents:
            for parent in self._parents:
                parent._get_roots_helper(result)
        else:
            result[id(self)] = self
        return result

    def parent_indices(self, parent):
        """
        Return a list of the indices where this tree occurs as a child
        of ``parent``.  If this child does not occur as a child of
        ``parent``, then the empty list is returned.  The following is
        always true::

          for parent_index in ptree.parent_indices(parent):
              parent[parent_index] is ptree
        """
        if parent not in self._parents:
            return []
        else:
            return [index for index, child in enumerate(parent) if child is self]

    def treepositions(self, root):
        """
        Return a list of all tree positions that can be used to reach
        this multi-parented tree starting from ``root``.  I.e., the
        following is always true::

          for treepos in ptree.treepositions(root):
              root[treepos] is ptree
        """
        if self is root:
            return [()]
        else:
            return [treepos + (index,) for parent in self._parents for treepos in parent.treepositions(root) for index, child in enumerate(parent) if child is self]

    def _delparent(self, child, index):
        assert isinstance(child, MultiParentedTree)
        assert self[index] is child
        assert len([p for p in child._parents if p is self]) == 1
        for i, c in enumerate(self):
            if c is child and i != index:
                break
        else:
            child._parents.remove(self)

    def _setparent(self, child, index, dry_run=False):
        if not isinstance(child, MultiParentedTree):
            raise TypeError('Can not insert a non-MultiParentedTree into a MultiParentedTree')
        if not dry_run:
            for parent in child._parents:
                if parent is self:
                    break
            else:
                child._parents.append(self)