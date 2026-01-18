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
class _NamedTupleCodec:
    """Codec for namedtuples.

  Encoding and decoding a namedtuple reconstructs a namedtuple with a different
  actual Python type, but with the same `typename` and `fields`.
  """

    def can_encode(self, pyobj):
        return _is_named_tuple(pyobj)

    def do_encode(self, named_tuple_value, encode_fn):
        encoded_named_tuple = struct_pb2.StructuredValue()
        encoded_named_tuple.named_tuple_value.CopyFrom(struct_pb2.NamedTupleValue())
        encoded_named_tuple.named_tuple_value.name = named_tuple_value.__class__.__name__
        for key in named_tuple_value._fields:
            pair = encoded_named_tuple.named_tuple_value.values.add()
            pair.key = key
            pair.value.CopyFrom(encode_fn(named_tuple_value._asdict()[key]))
        return encoded_named_tuple

    def can_decode(self, value):
        return value.HasField('named_tuple_value')

    def do_decode(self, value, decode_fn):
        key_value_pairs = value.named_tuple_value.values
        items = [(pair.key, decode_fn(pair.value)) for pair in key_value_pairs]
        named_tuple_type = collections.namedtuple(value.named_tuple_value.name, [item[0] for item in items])
        return named_tuple_type(**dict(items))