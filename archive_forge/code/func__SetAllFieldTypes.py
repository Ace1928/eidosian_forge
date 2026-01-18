import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def _SetAllFieldTypes(self, package, desc_proto, scope):
    """Sets all the descriptor's fields's types.

    This method also sets the containing types on any extensions.

    Args:
      package: The current package of desc_proto.
      desc_proto: The message descriptor to update.
      scope: Enclosing scope of available types.
    """
    package = _PrefixWithDot(package)
    main_desc = self._GetTypeFromScope(package, desc_proto.name, scope)
    if package == '.':
        nested_package = _PrefixWithDot(desc_proto.name)
    else:
        nested_package = '.'.join([package, desc_proto.name])
    for field_proto, field_desc in zip(desc_proto.field, main_desc.fields):
        self._SetFieldType(field_proto, field_desc, nested_package, scope)
    for extension_proto, extension_desc in zip(desc_proto.extension, main_desc.extensions):
        extension_desc.containing_type = self._GetTypeFromScope(nested_package, extension_proto.extendee, scope)
        self._SetFieldType(extension_proto, extension_desc, nested_package, scope)
    for nested_type in desc_proto.nested_type:
        self._SetAllFieldTypes(nested_package, nested_type, scope)