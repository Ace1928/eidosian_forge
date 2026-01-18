import calendar
import collections.abc
from datetime import datetime
from datetime import timedelta
from cloudsdk.google.protobuf.descriptor import FieldDescriptor
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