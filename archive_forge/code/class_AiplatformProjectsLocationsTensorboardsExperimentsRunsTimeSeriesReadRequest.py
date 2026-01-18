from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesReadRequest(_messages.Message):
    """A
  AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesReadRequest
  object.

  Fields:
    filter: Reads the TensorboardTimeSeries' data that match the filter
      expression.
    maxDataPoints: The maximum number of TensorboardTimeSeries' data to
      return. This value should be a positive integer. This value can be set
      to -1 to return all data.
    tensorboardTimeSeries: Required. The resource name of the
      TensorboardTimeSeries to read data from. Format: `projects/{project}/loc
      ations/{location}/tensorboards/{tensorboard}/experiments/{experiment}/ru
      ns/{run}/timeSeries/{time_series}`
  """
    filter = _messages.StringField(1)
    maxDataPoints = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    tensorboardTimeSeries = _messages.StringField(3, required=True)