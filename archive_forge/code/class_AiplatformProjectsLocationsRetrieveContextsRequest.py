from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsRetrieveContextsRequest(_messages.Message):
    """A AiplatformProjectsLocationsRetrieveContextsRequest object.

  Fields:
    googleCloudAiplatformV1beta1RetrieveContextsRequest: A
      GoogleCloudAiplatformV1beta1RetrieveContextsRequest resource to be
      passed as the request body.
    parent: Required. The resource name of the Location from which to retrieve
      RagContexts. The users must have permission to make a call in the
      project. Format: `projects/{project}/locations/{location}`.
  """
    googleCloudAiplatformV1beta1RetrieveContextsRequest = _messages.MessageField('GoogleCloudAiplatformV1beta1RetrieveContextsRequest', 1)
    parent = _messages.StringField(2, required=True)