from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ListTensorboardTimeSeriesResponse(_messages.Message):
    """Response message for TensorboardService.ListTensorboardTimeSeries.

  Fields:
    nextPageToken: A token, which can be sent as
      ListTensorboardTimeSeriesRequest.page_token to retrieve the next page.
      If this field is omitted, there are no subsequent pages.
    tensorboardTimeSeries: The TensorboardTimeSeries mathching the request.
  """
    nextPageToken = _messages.StringField(1)
    tensorboardTimeSeries = _messages.MessageField('GoogleCloudAiplatformV1TensorboardTimeSeries', 2, repeated=True)