from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ListDataItemsResponse(_messages.Message):
    """Response message for DatasetService.ListDataItems.

  Fields:
    dataItems: A list of DataItems that matches the specified filter in the
      request.
    nextPageToken: The standard List next-page token.
  """
    dataItems = _messages.MessageField('GoogleCloudAiplatformV1DataItem', 1, repeated=True)
    nextPageToken = _messages.StringField(2)