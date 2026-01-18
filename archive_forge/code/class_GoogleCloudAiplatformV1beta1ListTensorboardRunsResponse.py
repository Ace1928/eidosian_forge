from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListTensorboardRunsResponse(_messages.Message):
    """Response message for TensorboardService.ListTensorboardRuns.

  Fields:
    nextPageToken: A token, which can be sent as
      ListTensorboardRunsRequest.page_token to retrieve the next page. If this
      field is omitted, there are no subsequent pages.
    tensorboardRuns: The TensorboardRuns mathching the request.
  """
    nextPageToken = _messages.StringField(1)
    tensorboardRuns = _messages.MessageField('GoogleCloudAiplatformV1beta1TensorboardRun', 2, repeated=True)