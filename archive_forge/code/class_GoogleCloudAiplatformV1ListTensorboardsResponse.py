from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ListTensorboardsResponse(_messages.Message):
    """Response message for TensorboardService.ListTensorboards.

  Fields:
    nextPageToken: A token, which can be sent as
      ListTensorboardsRequest.page_token to retrieve the next page. If this
      field is omitted, there are no subsequent pages.
    tensorboards: The Tensorboards mathching the request.
  """
    nextPageToken = _messages.StringField(1)
    tensorboards = _messages.MessageField('GoogleCloudAiplatformV1Tensorboard', 2, repeated=True)