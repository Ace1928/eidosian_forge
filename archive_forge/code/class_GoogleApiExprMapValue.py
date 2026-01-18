from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprMapValue(_messages.Message):
    """A map. Wrapped in a message so 'not set' and empty can be
  differentiated, which is required for use in a 'oneof'.

  Fields:
    entries: The set of map entries. CEL has fewer restrictions on keys, so a
      protobuf map representation cannot be used.
  """
    entries = _messages.MessageField('GoogleApiExprMapValueEntry', 1, repeated=True)