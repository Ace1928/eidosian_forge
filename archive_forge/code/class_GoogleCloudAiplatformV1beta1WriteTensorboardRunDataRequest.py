from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1WriteTensorboardRunDataRequest(_messages.Message):
    """Request message for TensorboardService.WriteTensorboardRunData.

  Fields:
    tensorboardRun: Required. The resource name of the TensorboardRun to write
      data to. Format: `projects/{project}/locations/{location}/tensorboards/{
      tensorboard}/experiments/{experiment}/runs/{run}`
    timeSeriesData: Required. The TensorboardTimeSeries data to write. Values
      with in a time series are indexed by their step value. Repeated writes
      to the same step will overwrite the existing value for that step. The
      upper limit of data points per write request is 5000.
  """
    tensorboardRun = _messages.StringField(1)
    timeSeriesData = _messages.MessageField('GoogleCloudAiplatformV1beta1TimeSeriesData', 2, repeated=True)