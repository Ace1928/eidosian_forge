from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudNetworkconnectivityV1betaListCustomHardwareLinkAttachmentsResponse(_messages.Message):
    """Response for ListCustomHardwareLinkAttachments.

  Fields:
    customHardwareLinkAttachments: The list of CustomHardwareLinkAttachment
    nextPageToken: The next pagination token in the List response. It should
      be used as page_token for the following request. An empty value means no
      more result.
    unreachable: Locations that could not be reached.
  """
    customHardwareLinkAttachments = _messages.MessageField('GoogleCloudNetworkconnectivityV1betaCustomHardwareLinkAttachment', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)