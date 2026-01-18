import calendar
import collections.abc
from datetime import datetime
from datetime import timedelta
from cloudsdk.google.protobuf.descriptor import FieldDescriptor
def IsValidForDescriptor(self, message_descriptor):
    """Checks whether the FieldMask is valid for Message Descriptor."""
    for path in self.paths:
        if not _IsValidPath(message_descriptor, path):
            return False
    return True