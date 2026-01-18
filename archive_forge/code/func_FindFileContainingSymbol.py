import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def FindFileContainingSymbol(self, symbol):
    """Gets the FileDescriptor for the file containing the specified symbol.

    Args:
      symbol (str): The name of the symbol to search for.

    Returns:
      FileDescriptor: Descriptor for the file that contains the specified
      symbol.

    Raises:
      KeyError: if the file cannot be found in the pool.
    """
    symbol = _NormalizeFullyQualifiedName(symbol)
    try:
        return self._InternalFindFileContainingSymbol(symbol)
    except KeyError:
        pass
    try:
        self._FindFileContainingSymbolInDb(symbol)
        return self._InternalFindFileContainingSymbol(symbol)
    except KeyError:
        raise KeyError('Cannot find a file containing %s' % symbol)