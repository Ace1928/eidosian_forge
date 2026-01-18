from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReplaceAccessLevelsRequest(_messages.Message):
    """A request to replace all existing Access Levels in an Access Policy with
  the Access Levels provided. This is done atomically.

  Fields:
    accessLevels: Required. The desired Access Levels that should replace all
      existing Access Levels in the Access Policy.
    etag: Optional. The etag for the version of the Access Policy that this
      replace operation is to be performed on. If, at the time of replace, the
      etag for the Access Policy stored in Access Context Manager is different
      from the specified etag, then the replace operation will not be
      performed and the call will fail. This field is not required. If etag is
      not provided, the operation will be performed as if a valid etag is
      provided.
  """
    accessLevels = _messages.MessageField('AccessLevel', 1, repeated=True)
    etag = _messages.StringField(2)