from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def _ExtensionByteSize(self, partial):
    size = 0
    for extension, value in six.iteritems(self._extension_fields):
        ftype = extension.field_type
        tag_size = self.lengthVarInt64(extension.wire_tag)
        if ftype == TYPE_GROUP:
            tag_size *= 2
        if extension.is_repeated:
            size += tag_size * len(value)
            for single_value in value:
                size += self._FieldByteSize(ftype, single_value, partial)
        else:
            size += tag_size + self._FieldByteSize(ftype, value, partial)
    return size