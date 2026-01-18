import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
def _GetStructValue(struct_value):
    which = struct_value.WhichOneof('kind')
    if which == 'struct_value':
        return struct_value.struct_value
    elif which == 'null_value':
        return None
    elif which == 'number_value':
        return struct_value.number_value
    elif which == 'string_value':
        return struct_value.string_value
    elif which == 'bool_value':
        return struct_value.bool_value
    elif which == 'list_value':
        return struct_value.list_value
    elif which is None:
        raise ValueError('Value not set')