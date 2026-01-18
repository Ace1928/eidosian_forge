import calendar
import collections.abc
from datetime import datetime
from datetime import timedelta
from cloudsdk.google.protobuf.descriptor import FieldDescriptor
def MergeFromFieldMask(self, field_mask):
    """Merges a FieldMask to the tree."""
    for path in field_mask.paths:
        self.AddPath(path)