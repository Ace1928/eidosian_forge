from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListServiceProjectAttachmentsResponse(_messages.Message):
    """Response for ListServiceProjectAttachments.

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return.
    serviceProjectAttachments: List of service project attachments.
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    serviceProjectAttachments = _messages.MessageField('ServiceProjectAttachment', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)