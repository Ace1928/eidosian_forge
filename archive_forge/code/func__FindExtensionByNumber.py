from cloudsdk.google.protobuf.internal import type_checkers
from cloudsdk.google.protobuf.descriptor import FieldDescriptor
def _FindExtensionByNumber(self, number):
    """Tries to find a known extension with the field number.

    Args:
      number: Extension field number.

    Returns:
      Extension field descriptor.
    """
    return self._extended_message._extensions_by_number.get(number, None)