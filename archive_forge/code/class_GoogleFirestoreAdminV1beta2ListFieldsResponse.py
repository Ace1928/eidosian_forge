from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1beta2ListFieldsResponse(_messages.Message):
    """The response for FirestoreAdmin.ListFields.

  Fields:
    fields: The requested fields.
    nextPageToken: A page token that may be used to request another page of
      results. If blank, this is the last page.
  """
    fields = _messages.MessageField('GoogleFirestoreAdminV1beta2Field', 1, repeated=True)
    nextPageToken = _messages.StringField(2)