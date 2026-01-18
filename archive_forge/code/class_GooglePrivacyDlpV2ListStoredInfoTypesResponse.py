from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ListStoredInfoTypesResponse(_messages.Message):
    """Response message for ListStoredInfoTypes.

  Fields:
    nextPageToken: If the next page is available then the next page token to
      be used in the following ListStoredInfoTypes request.
    storedInfoTypes: List of storedInfoTypes, up to page_size in
      ListStoredInfoTypesRequest.
  """
    nextPageToken = _messages.StringField(1)
    storedInfoTypes = _messages.MessageField('GooglePrivacyDlpV2StoredInfoType', 2, repeated=True)