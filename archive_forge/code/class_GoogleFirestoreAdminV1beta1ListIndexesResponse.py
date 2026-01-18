from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1beta1ListIndexesResponse(_messages.Message):
    """The response for FirestoreAdmin.ListIndexes.

  Fields:
    indexes: The indexes.
    nextPageToken: The standard List next-page token.
  """
    indexes = _messages.MessageField('GoogleFirestoreAdminV1beta1Index', 1, repeated=True)
    nextPageToken = _messages.StringField(2)