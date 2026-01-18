from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListIndexesResponse(_messages.Message):
    """Response message for IndexService.ListIndexes.

  Fields:
    indexes: List of indexes in the requested page.
    nextPageToken: A token to retrieve next page of results. Pass to
      ListIndexesRequest.page_token to obtain that page.
  """
    indexes = _messages.MessageField('GoogleCloudAiplatformV1beta1Index', 1, repeated=True)
    nextPageToken = _messages.StringField(2)