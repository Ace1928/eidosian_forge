from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DatabasesListResponse(_messages.Message):
    """Database list response.

  Fields:
    items: List of database resources in the instance.
    kind: This is always `sql#databasesList`.
  """
    items = _messages.MessageField('Database', 1, repeated=True)
    kind = _messages.StringField(2)