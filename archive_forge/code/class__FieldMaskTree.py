import calendar
import collections.abc
from datetime import datetime
from datetime import timedelta
from cloudsdk.google.protobuf.descriptor import FieldDescriptor
class _FieldMaskTree(object):
    """Represents a FieldMask in a tree structure.

  For example, given a FieldMask "foo.bar,foo.baz,bar.baz",
  the FieldMaskTree will be:
      [_root] -+- foo -+- bar
            |       |
            |       +- baz
            |
            +- bar --- baz
  In the tree, each leaf node represents a field path.
  """
    __slots__ = ('_root',)

    def __init__(self, field_mask=None):
        """Initializes the tree by FieldMask."""
        self._root = {}
        if field_mask:
            self.MergeFromFieldMask(field_mask)

    def MergeFromFieldMask(self, field_mask):
        """Merges a FieldMask to the tree."""
        for path in field_mask.paths:
            self.AddPath(path)

    def AddPath(self, path):
        """Adds a field path into the tree.

    If the field path to add is a sub-path of an existing field path
    in the tree (i.e., a leaf node), it means the tree already matches
    the given path so nothing will be added to the tree. If the path
    matches an existing non-leaf node in the tree, that non-leaf node
    will be turned into a leaf node with all its children removed because
    the path matches all the node's children. Otherwise, a new path will
    be added.

    Args:
      path: The field path to add.
    """
        node = self._root
        for name in path.split('.'):
            if name not in node:
                node[name] = {}
            elif not node[name]:
                return
            node = node[name]
        node.clear()

    def ToFieldMask(self, field_mask):
        """Converts the tree to a FieldMask."""
        field_mask.Clear()
        _AddFieldPaths(self._root, '', field_mask)

    def IntersectPath(self, path, intersection):
        """Calculates the intersection part of a field path with this tree.

    Args:
      path: The field path to calculates.
      intersection: The out tree to record the intersection part.
    """
        node = self._root
        for name in path.split('.'):
            if name not in node:
                return
            elif not node[name]:
                intersection.AddPath(path)
                return
            node = node[name]
        intersection.AddLeafNodes(path, node)

    def AddLeafNodes(self, prefix, node):
        """Adds leaf nodes begin with prefix to this tree."""
        if not node:
            self.AddPath(prefix)
        for name in node:
            child_path = prefix + '.' + name
            self.AddLeafNodes(child_path, node[name])

    def MergeMessage(self, source, destination, replace_message, replace_repeated):
        """Merge all fields specified by this tree from source to destination."""
        _MergeMessage(self._root, source, destination, replace_message, replace_repeated)