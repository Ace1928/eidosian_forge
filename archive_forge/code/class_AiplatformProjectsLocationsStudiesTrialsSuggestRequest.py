from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsStudiesTrialsSuggestRequest(_messages.Message):
    """A AiplatformProjectsLocationsStudiesTrialsSuggestRequest object.

  Fields:
    googleCloudAiplatformV1SuggestTrialsRequest: A
      GoogleCloudAiplatformV1SuggestTrialsRequest resource to be passed as the
      request body.
    parent: Required. The project and location that the Study belongs to.
      Format: `projects/{project}/locations/{location}/studies/{study}`
  """
    googleCloudAiplatformV1SuggestTrialsRequest = _messages.MessageField('GoogleCloudAiplatformV1SuggestTrialsRequest', 1)
    parent = _messages.StringField(2, required=True)