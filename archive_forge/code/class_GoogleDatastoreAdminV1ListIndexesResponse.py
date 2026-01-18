from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDatastoreAdminV1ListIndexesResponse(_messages.Message):
    """The response for google.datastore.admin.v1.DatastoreAdmin.ListIndexes.

  Fields:
    indexes: The indexes.
    nextPageToken: The standard List next-page token.
  """
    indexes = _messages.MessageField('GoogleDatastoreAdminV1Index', 1, repeated=True)
    nextPageToken = _messages.StringField(2)