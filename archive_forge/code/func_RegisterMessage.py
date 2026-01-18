from cloudsdk.google.protobuf.internal import api_implementation
from cloudsdk.google.protobuf import descriptor_pool
from cloudsdk.google.protobuf import message_factory
def RegisterMessage(self, message):
    """Registers the given message type in the local database.

    Calls to GetSymbol() and GetMessages() will return messages registered here.

    Args:
      message: A :class:`google.protobuf.message.Message` subclass (or
        instance); its descriptor will be registered.

    Returns:
      The provided message.
    """
    desc = message.DESCRIPTOR
    self._classes[desc] = message
    self.RegisterMessageDescriptor(desc)
    return message