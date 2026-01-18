from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
@staticmethod
def _RegisterExtension(cls, extension, composite_cls=None):
    extension.containing_cls = cls
    extension.composite_cls = composite_cls
    if composite_cls is not None:
        extension.message_name = composite_cls._PROTO_DESCRIPTOR_NAME
    actual_handle = cls._extensions_by_field_number.setdefault(extension.number, extension)
    if actual_handle is not extension:
        raise AssertionError('Extensions "%s" and "%s" both try to extend message type "%s" with field number %d.' % (extension.full_name, actual_handle.full_name, cls.__name__, extension.number))