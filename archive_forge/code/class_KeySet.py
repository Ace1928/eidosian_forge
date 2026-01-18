from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KeySet(_messages.Message):
    """`KeySet` defines a collection of Cloud Spanner keys and/or key ranges.
  All the keys are expected to be in the same table or index. The keys need
  not be sorted in any particular way. If the same key is specified multiple
  times in the set (for example if two ranges, two keys, or a key and a range
  overlap), Cloud Spanner behaves as if the key were only specified once.

  Messages:
    KeysValueListEntry: Single entry in a KeysValue.

  Fields:
    all: For convenience `all` can be set to `true` to indicate that this
      `KeySet` matches all keys in the table or index. Note that any keys
      specified in `keys` or `ranges` are only yielded once.
    keys: A list of specific keys. Entries in `keys` should have exactly as
      many elements as there are columns in the primary or index key with
      which this `KeySet` is used. Individual key values are encoded as
      described here.
    ranges: A list of key ranges. See KeyRange for more information about key
      range specifications.
  """

    class KeysValueListEntry(_messages.Message):
        """Single entry in a KeysValue.

    Fields:
      entry: A extra_types.JsonValue attribute.
    """
        entry = _messages.MessageField('extra_types.JsonValue', 1, repeated=True)
    all = _messages.BooleanField(1)
    keys = _messages.MessageField('KeysValueListEntry', 2, repeated=True)
    ranges = _messages.MessageField('KeyRange', 3, repeated=True)