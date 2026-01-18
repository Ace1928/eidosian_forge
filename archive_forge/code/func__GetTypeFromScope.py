import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def _GetTypeFromScope(self, package, type_name, scope):
    """Finds a given type name in the current scope.

    Args:
      package: The package the proto should be located in.
      type_name: The name of the type to be found in the scope.
      scope: Dict mapping short and full symbols to message and enum types.

    Returns:
      The descriptor for the requested type.
    """
    if type_name not in scope:
        components = _PrefixWithDot(package).split('.')
        while components:
            possible_match = '.'.join(components + [type_name])
            if possible_match in scope:
                type_name = possible_match
                break
            else:
                components.pop(-1)
    return scope[type_name]