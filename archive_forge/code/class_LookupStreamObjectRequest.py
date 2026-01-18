from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LookupStreamObjectRequest(_messages.Message):
    """Request for looking up a specific stream object by its source object
  identifier.

  Fields:
    sourceObjectIdentifier: Required. The source object identifier which maps
      to the stream object.
  """
    sourceObjectIdentifier = _messages.MessageField('SourceObjectIdentifier', 1)