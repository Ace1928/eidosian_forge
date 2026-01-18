from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsModelsEvaluationsImportRequest(_messages.Message):
    """A AiplatformProjectsLocationsModelsEvaluationsImportRequest object.

  Fields:
    googleCloudAiplatformV1ImportModelEvaluationRequest: A
      GoogleCloudAiplatformV1ImportModelEvaluationRequest resource to be
      passed as the request body.
    parent: Required. The name of the parent model resource. Format:
      `projects/{project}/locations/{location}/models/{model}`
  """
    googleCloudAiplatformV1ImportModelEvaluationRequest = _messages.MessageField('GoogleCloudAiplatformV1ImportModelEvaluationRequest', 1)
    parent = _messages.StringField(2, required=True)