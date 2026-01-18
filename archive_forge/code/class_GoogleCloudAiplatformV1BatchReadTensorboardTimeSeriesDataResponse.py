from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1BatchReadTensorboardTimeSeriesDataResponse(_messages.Message):
    """Response message for
  TensorboardService.BatchReadTensorboardTimeSeriesData.

  Fields:
    timeSeriesData: The returned time series data.
  """
    timeSeriesData = _messages.MessageField('GoogleCloudAiplatformV1TimeSeriesData', 1, repeated=True)