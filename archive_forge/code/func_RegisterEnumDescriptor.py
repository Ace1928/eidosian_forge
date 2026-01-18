from cloudsdk.google.protobuf.internal import api_implementation
from cloudsdk.google.protobuf import descriptor_pool
from cloudsdk.google.protobuf import message_factory
def RegisterEnumDescriptor(self, enum_descriptor):
    """Registers the given enum descriptor in the local database.

    Args:
      enum_descriptor (EnumDescriptor): The enum descriptor to register.

    Returns:
      EnumDescriptor: The provided descriptor.
    """
    if api_implementation.Type() == 'python':
        self.pool._AddEnumDescriptor(enum_descriptor)
    return enum_descriptor