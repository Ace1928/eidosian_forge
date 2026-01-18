from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsExperimentsCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsTensorboardsExperimentsCreateRequest
  object.

  Fields:
    googleCloudAiplatformV1TensorboardExperiment: A
      GoogleCloudAiplatformV1TensorboardExperiment resource to be passed as
      the request body.
    parent: Required. The resource name of the Tensorboard to create the
      TensorboardExperiment in. Format:
      `projects/{project}/locations/{location}/tensorboards/{tensorboard}`
    tensorboardExperimentId: Required. The ID to use for the Tensorboard
      experiment, which becomes the final component of the Tensorboard
      experiment's resource name. This value should be 1-128 characters, and
      valid characters are `/a-z-/`.
  """
    googleCloudAiplatformV1TensorboardExperiment = _messages.MessageField('GoogleCloudAiplatformV1TensorboardExperiment', 1)
    parent = _messages.StringField(2, required=True)
    tensorboardExperimentId = _messages.StringField(3)