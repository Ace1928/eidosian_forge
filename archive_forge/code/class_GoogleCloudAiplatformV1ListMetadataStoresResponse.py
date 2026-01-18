from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ListMetadataStoresResponse(_messages.Message):
    """Response message for MetadataService.ListMetadataStores.

  Fields:
    metadataStores: The MetadataStores found for the Location.
    nextPageToken: A token, which can be sent as
      ListMetadataStoresRequest.page_token to retrieve the next page. If this
      field is not populated, there are no subsequent pages.
  """
    metadataStores = _messages.MessageField('GoogleCloudAiplatformV1MetadataStore', 1, repeated=True)
    nextPageToken = _messages.StringField(2)