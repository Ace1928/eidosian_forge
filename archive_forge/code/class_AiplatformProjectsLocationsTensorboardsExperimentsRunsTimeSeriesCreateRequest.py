from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesCreate
  Request object.

  Fields:
    googleCloudAiplatformV1TensorboardTimeSeries: A
      GoogleCloudAiplatformV1TensorboardTimeSeries resource to be passed as
      the request body.
    parent: Required. The resource name of the TensorboardRun to create the
      TensorboardTimeSeries in. Format: `projects/{project}/locations/{locatio
      n}/tensorboards/{tensorboard}/experiments/{experiment}/runs/{run}`
    tensorboardTimeSeriesId: Optional. The user specified unique ID to use for
      the TensorboardTimeSeries, which becomes the final component of the
      TensorboardTimeSeries's resource name. This value should match
      "a-z0-9{0, 127}"
  """
    googleCloudAiplatformV1TensorboardTimeSeries = _messages.MessageField('GoogleCloudAiplatformV1TensorboardTimeSeries', 1)
    parent = _messages.StringField(2, required=True)
    tensorboardTimeSeriesId = _messages.StringField(3)