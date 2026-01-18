import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
def Is(self, descriptor):
    """Checks if this Any represents the given protobuf type."""
    return '/' in self.type_url and self.TypeName() == descriptor.full_name