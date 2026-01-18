from cloudsdk.google.protobuf.internal import api_implementation
from cloudsdk.google.protobuf import descriptor_pool
from cloudsdk.google.protobuf import message_factory
def RegisterFileDescriptor(self, file_descriptor):
    """Registers the given file descriptor in the local database.

    Args:
      file_descriptor (FileDescriptor): The file descriptor to register.
    """
    if api_implementation.Type() == 'python':
        self.pool._InternalAddFileDescriptor(file_descriptor)