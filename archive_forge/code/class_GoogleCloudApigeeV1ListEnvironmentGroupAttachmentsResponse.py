from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListEnvironmentGroupAttachmentsResponse(_messages.Message):
    """Response for ListEnvironmentGroupAttachments.

  Fields:
    environmentGroupAttachments: EnvironmentGroupAttachments for the specified
      environment group.
    nextPageToken: Page token that you can include in a
      ListEnvironmentGroupAttachments request to retrieve the next page. If
      omitted, no subsequent pages exist.
  """
    environmentGroupAttachments = _messages.MessageField('GoogleCloudApigeeV1EnvironmentGroupAttachment', 1, repeated=True)
    nextPageToken = _messages.StringField(2)