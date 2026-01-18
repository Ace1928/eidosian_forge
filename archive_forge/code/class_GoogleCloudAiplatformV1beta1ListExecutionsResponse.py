from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListExecutionsResponse(_messages.Message):
    """Response message for MetadataService.ListExecutions.

  Fields:
    executions: The Executions retrieved from the MetadataStore.
    nextPageToken: A token, which can be sent as
      ListExecutionsRequest.page_token to retrieve the next page. If this
      field is not populated, there are no subsequent pages.
  """
    executions = _messages.MessageField('GoogleCloudAiplatformV1beta1Execution', 1, repeated=True)
    nextPageToken = _messages.StringField(2)