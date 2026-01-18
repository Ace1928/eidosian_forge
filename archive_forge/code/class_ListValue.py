import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
class ListValue(object):
    """Class for ListValue message type."""
    __slots__ = ()

    def __len__(self):
        return len(self.values)

    def append(self, value):
        _SetStructValue(self.values.add(), value)

    def extend(self, elem_seq):
        for value in elem_seq:
            self.append(value)

    def __getitem__(self, index):
        """Retrieves item by the specified index."""
        return _GetStructValue(self.values.__getitem__(index))

    def __setitem__(self, index, value):
        _SetStructValue(self.values.__getitem__(index), value)

    def __delitem__(self, key):
        del self.values[key]

    def items(self):
        for i in range(len(self)):
            yield self[i]

    def add_struct(self):
        """Appends and returns a struct value as the next value in the list."""
        struct_value = self.values.add().struct_value
        struct_value.Clear()
        return struct_value

    def add_list(self):
        """Appends and returns a list value as the next value in the list."""
        list_value = self.values.add().list_value
        list_value.Clear()
        return list_value