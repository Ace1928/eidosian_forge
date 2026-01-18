import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
class _NestedDescriptorBase(DescriptorBase):
    """Common class for descriptors that can be nested."""

    def __init__(self, options, options_class_name, name, full_name, file, containing_type, serialized_start=None, serialized_end=None, serialized_options=None):
        """Constructor.

    Args:
      options: Protocol message options or None to use default message options.
      options_class_name (str): The class name of the above options.
      name (str): Name of this protocol message type.
      full_name (str): Fully-qualified name of this protocol message type, which
        will include protocol "package" name and the name of any enclosing
        types.
      containing_type: if provided, this is a nested descriptor, with this
        descriptor as parent, otherwise None.
      serialized_start: The start index (inclusive) in block in the
        file.serialized_pb that describes this descriptor.
      serialized_end: The end index (exclusive) in block in the
        file.serialized_pb that describes this descriptor.
      serialized_options: Protocol message serialized options or None.
    """
        super(_NestedDescriptorBase, self).__init__(file, options, serialized_options, options_class_name)
        self.name = name
        self.full_name = full_name
        self.containing_type = containing_type
        self._serialized_start = serialized_start
        self._serialized_end = serialized_end

    def CopyToProto(self, proto):
        """Copies this to the matching proto in descriptor_pb2.

    Args:
      proto: An empty proto instance from descriptor_pb2.

    Raises:
      Error: If self couldn't be serialized, due to to few constructor
        arguments.
    """
        if self.file is not None and self._serialized_start is not None and (self._serialized_end is not None):
            proto.ParseFromString(self.file.serialized_pb[self._serialized_start:self._serialized_end])
        else:
            raise Error('Descriptor does not contain serialization.')