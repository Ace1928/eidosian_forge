from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1CreateTensorboardTimeSeriesRequest(_messages.Message):
    """Request message for TensorboardService.CreateTensorboardTimeSeries.

  Fields:
    parent: Required. The resource name of the TensorboardRun to create the
      TensorboardTimeSeries in. Format: `projects/{project}/locations/{locatio
      n}/tensorboards/{tensorboard}/experiments/{experiment}/runs/{run}`
    tensorboardTimeSeries: Required. The TensorboardTimeSeries to create.
    tensorboardTimeSeriesId: Optional. The user specified unique ID to use for
      the TensorboardTimeSeries, which becomes the final component of the
      TensorboardTimeSeries's resource name. This value should match
      "a-z0-9{0, 127}"
  """
    parent = _messages.StringField(1)
    tensorboardTimeSeries = _messages.MessageField('GoogleCloudAiplatformV1beta1TensorboardTimeSeries', 2)
    tensorboardTimeSeriesId = _messages.StringField(3)