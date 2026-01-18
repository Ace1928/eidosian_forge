import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def _InternalFindFileContainingSymbol(self, symbol):
    """Gets the already built FileDescriptor containing the specified symbol.

    Args:
      symbol (str): The name of the symbol to search for.

    Returns:
      FileDescriptor: Descriptor for the file that contains the specified
      symbol.

    Raises:
      KeyError: if the file cannot be found in the pool.
    """
    try:
        return self._descriptors[symbol].file
    except KeyError:
        pass
    try:
        return self._enum_descriptors[symbol].file
    except KeyError:
        pass
    try:
        return self._service_descriptors[symbol].file
    except KeyError:
        pass
    try:
        return self._top_enum_values[symbol].type.file
    except KeyError:
        pass
    try:
        return self._toplevel_extensions[symbol].file
    except KeyError:
        pass
    top_name, _, sub_name = symbol.rpartition('.')
    try:
        message = self.FindMessageTypeByName(top_name)
        assert sub_name in message.extensions_by_name or sub_name in message.fields_by_name or sub_name in message.enum_values_by_name
        return message.file
    except (KeyError, AssertionError):
        raise KeyError('Cannot find a file containing %s' % symbol)