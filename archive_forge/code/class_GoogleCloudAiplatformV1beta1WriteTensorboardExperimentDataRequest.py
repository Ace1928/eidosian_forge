from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1WriteTensorboardExperimentDataRequest(_messages.Message):
    """Request message for TensorboardService.WriteTensorboardExperimentData.

  Fields:
    writeRunDataRequests: Required. Requests containing per-run
      TensorboardTimeSeries data to write.
  """
    writeRunDataRequests = _messages.MessageField('GoogleCloudAiplatformV1beta1WriteTensorboardRunDataRequest', 1, repeated=True)