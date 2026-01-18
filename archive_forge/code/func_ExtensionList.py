from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def ExtensionList(self, extension):
    """Returns a mutable list of extensions.

    Raises:
      TypeError if the extension is not repeated.
    """
    self._VerifyExtensionIdentifier(extension)
    if not extension.is_repeated:
        raise TypeError('ExtensionList() cannot be applied to "%s", because it is not a repeated extension.' % extension.full_name)
    if extension in self._extension_fields:
        return self._extension_fields[extension]
    result = []
    self._extension_fields[extension] = result
    return result