from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsStudiesTrialsListOptimalTrialsRequest(_messages.Message):
    """A AiplatformProjectsLocationsStudiesTrialsListOptimalTrialsRequest
  object.

  Fields:
    googleCloudAiplatformV1ListOptimalTrialsRequest: A
      GoogleCloudAiplatformV1ListOptimalTrialsRequest resource to be passed as
      the request body.
    parent: Required. The name of the Study that the optimal Trial belongs to.
  """
    googleCloudAiplatformV1ListOptimalTrialsRequest = _messages.MessageField('GoogleCloudAiplatformV1ListOptimalTrialsRequest', 1)
    parent = _messages.StringField(2, required=True)