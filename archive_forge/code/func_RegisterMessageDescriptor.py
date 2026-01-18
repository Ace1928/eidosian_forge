from cloudsdk.google.protobuf.internal import api_implementation
from cloudsdk.google.protobuf import descriptor_pool
from cloudsdk.google.protobuf import message_factory
def RegisterMessageDescriptor(self, message_descriptor):
    """Registers the given message descriptor in the local database.

    Args:
      message_descriptor (Descriptor): the message descriptor to add.
    """
    if api_implementation.Type() == 'python':
        self.pool._AddDescriptor(message_descriptor)