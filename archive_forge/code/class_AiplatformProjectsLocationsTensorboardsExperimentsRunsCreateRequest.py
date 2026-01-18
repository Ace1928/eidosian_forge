from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsExperimentsRunsCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsTensorboardsExperimentsRunsCreateRequest
  object.

  Fields:
    googleCloudAiplatformV1TensorboardRun: A
      GoogleCloudAiplatformV1TensorboardRun resource to be passed as the
      request body.
    parent: Required. The resource name of the TensorboardExperiment to create
      the TensorboardRun in. Format: `projects/{project}/locations/{location}/
      tensorboards/{tensorboard}/experiments/{experiment}`
    tensorboardRunId: Required. The ID to use for the Tensorboard run, which
      becomes the final component of the Tensorboard run's resource name. This
      value should be 1-128 characters, and valid characters are `/a-z-/`.
  """
    googleCloudAiplatformV1TensorboardRun = _messages.MessageField('GoogleCloudAiplatformV1TensorboardRun', 1)
    parent = _messages.StringField(2, required=True)
    tensorboardRunId = _messages.StringField(3)