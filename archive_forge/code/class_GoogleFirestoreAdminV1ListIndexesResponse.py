from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1ListIndexesResponse(_messages.Message):
    """The response for FirestoreAdmin.ListIndexes.

  Fields:
    indexes: The requested indexes.
    nextPageToken: A page token that may be used to request another page of
      results. If blank, this is the last page.
  """
    indexes = _messages.MessageField('GoogleFirestoreAdminV1Index', 1, repeated=True)
    nextPageToken = _messages.StringField(2)