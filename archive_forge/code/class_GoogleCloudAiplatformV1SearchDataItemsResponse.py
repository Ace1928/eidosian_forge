from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SearchDataItemsResponse(_messages.Message):
    """Response message for DatasetService.SearchDataItems.

  Fields:
    dataItemViews: The DataItemViews read.
    nextPageToken: A token to retrieve next page of results. Pass to
      SearchDataItemsRequest.page_token to obtain that page.
  """
    dataItemViews = _messages.MessageField('GoogleCloudAiplatformV1DataItemView', 1, repeated=True)
    nextPageToken = _messages.StringField(2)