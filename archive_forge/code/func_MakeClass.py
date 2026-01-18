from cloudsdk.google.protobuf import message_factory
from cloudsdk.google.protobuf import symbol_database
def MakeClass(descriptor):
    """Construct a class object for a protobuf described by descriptor.

  DEPRECATED: use MessageFactory.GetPrototype() instead.

  Args:
    descriptor: A descriptor.Descriptor object describing the protobuf.
  Returns:
    The Message class object described by the descriptor.
  """
    return symbol_database.Default().GetPrototype(descriptor)