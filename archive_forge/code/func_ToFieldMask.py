import calendar
import collections.abc
from datetime import datetime
from datetime import timedelta
from cloudsdk.google.protobuf.descriptor import FieldDescriptor
def ToFieldMask(self, field_mask):
    """Converts the tree to a FieldMask."""
    field_mask.Clear()
    _AddFieldPaths(self._root, '', field_mask)