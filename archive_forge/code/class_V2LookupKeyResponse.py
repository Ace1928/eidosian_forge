from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V2LookupKeyResponse(_messages.Message):
    """Response message for `LookupKey` method.

  Fields:
    name: The resource name of the API key. If the API key has been purged,
      resource name is empty.
    parent: The project that owns the key with the value specified in the
      request.
  """
    name = _messages.StringField(1)
    parent = _messages.StringField(2)