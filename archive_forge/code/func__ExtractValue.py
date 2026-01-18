from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import six
def _ExtractValue(proto):
    if proto.object_value:
        return _ExtractDecoratedObject(proto)
    if proto.array_value:
        return [_ExtractValue(v) for v in proto.array_value.entries]
    if proto.string_value:
        return proto.string_value
    return 'No decoding provided for: {0}'.format(proto)