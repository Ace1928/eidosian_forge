from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CacheInvalidationRule(_messages.Message):
    """A CacheInvalidationRule object.

  Fields:
    host: If set, this invalidation rule will only apply to requests with a
      Host header matching host.
    path: A string attribute.
  """
    host = _messages.StringField(1)
    path = _messages.StringField(2)