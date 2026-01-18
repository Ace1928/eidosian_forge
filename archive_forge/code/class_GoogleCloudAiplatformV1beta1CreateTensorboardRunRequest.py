from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1CreateTensorboardRunRequest(_messages.Message):
    """Request message for TensorboardService.CreateTensorboardRun.

  Fields:
    parent: Required. The resource name of the TensorboardExperiment to create
      the TensorboardRun in. Format: `projects/{project}/locations/{location}/
      tensorboards/{tensorboard}/experiments/{experiment}`
    tensorboardRun: Required. The TensorboardRun to create.
    tensorboardRunId: Required. The ID to use for the Tensorboard run, which
      becomes the final component of the Tensorboard run's resource name. This
      value should be 1-128 characters, and valid characters are `/a-z-/`.
  """
    parent = _messages.StringField(1)
    tensorboardRun = _messages.MessageField('GoogleCloudAiplatformV1beta1TensorboardRun', 2)
    tensorboardRunId = _messages.StringField(3)