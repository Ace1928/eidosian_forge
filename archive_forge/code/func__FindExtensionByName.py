from cloudsdk.google.protobuf.internal import type_checkers
from cloudsdk.google.protobuf.descriptor import FieldDescriptor
def _FindExtensionByName(self, name):
    """Tries to find a known extension with the specified name.

    Args:
      name: Extension full name.

    Returns:
      Extension field descriptor.
    """
    return self._extended_message._extensions_by_name.get(name, None)