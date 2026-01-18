from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def MutableExtension(self, extension, index=None):
    """Gets a mutable reference of a message type extension.

    For repeated extension, index must be specified, and only one element will
    be returned. For optional extension, if the extension does not exist, a new
    message will be created and set in parent message.

    Args:
      extension: The ExtensionIdentifier for the extension.
      index: The index of element to mutate in a repeated field. Only needed if
          the extension is repeated.

    Returns:
      The mutable message reference.

    Raises:
      TypeError if non-message type extension is given.
    """
    self._VerifyExtensionIdentifier(extension)
    if extension.composite_cls is None:
        raise TypeError('MutableExtension() cannot be applied to "%s", because it is not a composite type.' % extension.full_name)
    if extension.is_repeated:
        if index is None:
            raise TypeError('MutableExtension(extension, index) for repeated extension takes exactly 2 arguments: (1 given)')
        return self.GetExtension(extension, index)
    if extension in self._extension_fields:
        return self._extension_fields[extension]
    else:
        result = extension.composite_cls()
        self._extension_fields[extension] = result
        return result