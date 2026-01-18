import warnings
from google.protobuf.internal import api_implementation
from google.protobuf import descriptor_pool
from google.protobuf import message
def GetMessageClass(descriptor):
    """Obtains a proto2 message class based on the passed in descriptor.

  Passing a descriptor with a fully qualified name matching a previous
  invocation will cause the same class to be returned.

  Args:
    descriptor: The descriptor to build from.

  Returns:
    A class describing the passed in descriptor.
  """
    concrete_class = getattr(descriptor, '_concrete_class', None)
    if concrete_class:
        return concrete_class
    return _InternalCreateMessageClass(descriptor)