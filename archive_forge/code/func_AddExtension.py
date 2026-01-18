from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def AddExtension(self, extension, value=None):
    """Appends a new element into a repeated extension.

    Arg varies according to the extension field type:
    - Scalar/String:
      message.AddExtension(extension, value)
    - Message:
      mutable_message = AddExtension(extension)

    Args:
      extension: The ExtensionIdentifier for the extension.
      value: The value of the extension if the extension is scalar/string type.
          The value must NOT be set for message type extensions; set values on
          the returned message object instead.

    Returns:
      A mutable new message if it's a message type extension, or None otherwise.

    Raises:
      TypeError if the extension is not repeated, or value is given for message
      type extensions.
    """
    self._VerifyExtensionIdentifier(extension)
    if not extension.is_repeated:
        raise TypeError('AddExtension() cannot be applied to "%s", because it is not a repeated extension.' % extension.full_name)
    if extension in self._extension_fields:
        field = self._extension_fields[extension]
    else:
        field = []
        self._extension_fields[extension] = field
    if extension.composite_cls:
        if value is not None:
            raise TypeError('value must not be set in AddExtension() for "%s", because it is a message type extension. Set values on the returned message instead.' % extension.full_name)
        msg = extension.composite_cls()
        field.append(msg)
        return msg
    field.append(value)