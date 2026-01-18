from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesListRequest(_messages.Message):
    """A
  AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesListRequest
  object.

  Fields:
    filter: Lists the TensorboardTimeSeries that match the filter expression.
    orderBy: Field to use to sort the list.
    pageSize: The maximum number of TensorboardTimeSeries to return. The
      service may return fewer than this value. If unspecified, at most 50
      TensorboardTimeSeries are returned. The maximum value is 1000; values
      above 1000 are coerced to 1000.
    pageToken: A page token, received from a previous
      TensorboardService.ListTensorboardTimeSeries call. Provide this to
      retrieve the subsequent page. When paginating, all other parameters
      provided to TensorboardService.ListTensorboardTimeSeries must match the
      call that provided the page token.
    parent: Required. The resource name of the TensorboardRun to list
      TensorboardTimeSeries. Format: `projects/{project}/locations/{location}/
      tensorboards/{tensorboard}/experiments/{experiment}/runs/{run}`
    readMask: Mask specifying which fields to read.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
    readMask = _messages.StringField(6)