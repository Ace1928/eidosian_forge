from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def SetExtension(self, extension, *args):
    """Sets the extension value for a certain scalar type extension.

    Arg varies according to extension type:
    - Singular:
      message.SetExtension(extension, value)
    - Repeated:
      message.SetExtension(extension, index, value)
    where
      extension: The ExtensionIdentifier for the extension.
      index: The index of element to set in a repeated field. Only needed if
          the extension is repeated.
      value: The value to set.

    Raises:
      TypeError if a message type extension is given.
    """
    self._VerifyExtensionIdentifier(extension)
    if extension.composite_cls:
        raise TypeError('Cannot assign to extension "%s" because it is a composite type.' % extension.full_name)
    if extension.is_repeated:
        try:
            index, value = args
        except ValueError:
            raise TypeError('SetExtension(extension, index, value) for repeated extension takes exactly 4 arguments: (%d given)' % (len(args) + 2))
        self._extension_fields[extension][index] = value
    else:
        try:
            value, = args
        except ValueError:
            raise TypeError('SetExtension(extension, value) for singular extension takes exactly 3 arguments: (%d given)' % (len(args) + 2))
        self._extension_fields[extension] = value