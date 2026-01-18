from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1BatchCreateTensorboardRunsResponse(_messages.Message):
    """Response message for TensorboardService.BatchCreateTensorboardRuns.

  Fields:
    tensorboardRuns: The created TensorboardRuns.
  """
    tensorboardRuns = _messages.MessageField('GoogleCloudAiplatformV1beta1TensorboardRun', 1, repeated=True)