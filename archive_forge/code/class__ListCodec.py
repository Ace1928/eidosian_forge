import collections
import functools
import warnings
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class _ListCodec:
    """Codec for lists."""

    def can_encode(self, pyobj):
        return isinstance(pyobj, list)

    def do_encode(self, list_value, encode_fn):
        encoded_list = struct_pb2.StructuredValue()
        encoded_list.list_value.CopyFrom(struct_pb2.ListValue())
        for element in list_value:
            encoded_list.list_value.values.add().CopyFrom(encode_fn(element))
        return encoded_list

    def can_decode(self, value):
        return value.HasField('list_value')

    def do_decode(self, value, decode_fn):
        return [decode_fn(element) for element in value.list_value.values]