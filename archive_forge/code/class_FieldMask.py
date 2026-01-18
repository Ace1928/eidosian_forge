import calendar
import collections.abc
from datetime import datetime
from datetime import timedelta
from cloudsdk.google.protobuf.descriptor import FieldDescriptor
class FieldMask(object):
    """Class for FieldMask message type."""
    __slots__ = ()

    def ToJsonString(self):
        """Converts FieldMask to string according to proto3 JSON spec."""
        camelcase_paths = []
        for path in self.paths:
            camelcase_paths.append(_SnakeCaseToCamelCase(path))
        return ','.join(camelcase_paths)

    def FromJsonString(self, value):
        """Converts string to FieldMask according to proto3 JSON spec."""
        if not isinstance(value, str):
            raise ValueError('FieldMask JSON value not a string: {!r}'.format(value))
        self.Clear()
        if value:
            for path in value.split(','):
                self.paths.append(_CamelCaseToSnakeCase(path))

    def IsValidForDescriptor(self, message_descriptor):
        """Checks whether the FieldMask is valid for Message Descriptor."""
        for path in self.paths:
            if not _IsValidPath(message_descriptor, path):
                return False
        return True

    def AllFieldsFromDescriptor(self, message_descriptor):
        """Gets all direct fields of Message Descriptor to FieldMask."""
        self.Clear()
        for field in message_descriptor.fields:
            self.paths.append(field.name)

    def CanonicalFormFromMask(self, mask):
        """Converts a FieldMask to the canonical form.

    Removes paths that are covered by another path. For example,
    "foo.bar" is covered by "foo" and will be removed if "foo"
    is also in the FieldMask. Then sorts all paths in alphabetical order.

    Args:
      mask: The original FieldMask to be converted.
    """
        tree = _FieldMaskTree(mask)
        tree.ToFieldMask(self)

    def Union(self, mask1, mask2):
        """Merges mask1 and mask2 into this FieldMask."""
        _CheckFieldMaskMessage(mask1)
        _CheckFieldMaskMessage(mask2)
        tree = _FieldMaskTree(mask1)
        tree.MergeFromFieldMask(mask2)
        tree.ToFieldMask(self)

    def Intersect(self, mask1, mask2):
        """Intersects mask1 and mask2 into this FieldMask."""
        _CheckFieldMaskMessage(mask1)
        _CheckFieldMaskMessage(mask2)
        tree = _FieldMaskTree(mask1)
        intersection = _FieldMaskTree()
        for path in mask2.paths:
            tree.IntersectPath(path, intersection)
        intersection.ToFieldMask(self)

    def MergeMessage(self, source, destination, replace_message_field=False, replace_repeated_field=False):
        """Merges fields specified in FieldMask from source to destination.

    Args:
      source: Source message.
      destination: The destination message to be merged into.
      replace_message_field: Replace message field if True. Merge message
          field if False.
      replace_repeated_field: Replace repeated field if True. Append
          elements of repeated field if False.
    """
        tree = _FieldMaskTree(self)
        tree.MergeMessage(source, destination, replace_message_field, replace_repeated_field)