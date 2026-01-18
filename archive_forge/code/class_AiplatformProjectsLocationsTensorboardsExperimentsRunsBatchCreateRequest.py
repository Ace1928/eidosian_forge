from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsExperimentsRunsBatchCreateRequest(_messages.Message):
    """A
  AiplatformProjectsLocationsTensorboardsExperimentsRunsBatchCreateRequest
  object.

  Fields:
    googleCloudAiplatformV1BatchCreateTensorboardRunsRequest: A
      GoogleCloudAiplatformV1BatchCreateTensorboardRunsRequest resource to be
      passed as the request body.
    parent: Required. The resource name of the TensorboardExperiment to create
      the TensorboardRuns in. Format: `projects/{project}/locations/{location}
      /tensorboards/{tensorboard}/experiments/{experiment}` The parent field
      in the CreateTensorboardRunRequest messages must match this field.
  """
    googleCloudAiplatformV1BatchCreateTensorboardRunsRequest = _messages.MessageField('GoogleCloudAiplatformV1BatchCreateTensorboardRunsRequest', 1)
    parent = _messages.StringField(2, required=True)