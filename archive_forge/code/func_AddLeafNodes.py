import calendar
import collections.abc
from datetime import datetime
from datetime import timedelta
from cloudsdk.google.protobuf.descriptor import FieldDescriptor
def AddLeafNodes(self, prefix, node):
    """Adds leaf nodes begin with prefix to this tree."""
    if not node:
        self.AddPath(prefix)
    for name in node:
        child_path = prefix + '.' + name
        self.AddLeafNodes(child_path, node[name])