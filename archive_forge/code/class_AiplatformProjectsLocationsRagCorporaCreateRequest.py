from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsRagCorporaCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsRagCorporaCreateRequest object.

  Fields:
    googleCloudAiplatformV1beta1RagCorpus: A
      GoogleCloudAiplatformV1beta1RagCorpus resource to be passed as the
      request body.
    parent: Required. The resource name of the Location to create the
      RagCorpus in. Format: `projects/{project}/locations/{location}`
  """
    googleCloudAiplatformV1beta1RagCorpus = _messages.MessageField('GoogleCloudAiplatformV1beta1RagCorpus', 1)
    parent = _messages.StringField(2, required=True)