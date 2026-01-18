from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresExecutionsAddExecutionEventsRequest(_messages.Message):
    """A
  AiplatformProjectsLocationsMetadataStoresExecutionsAddExecutionEventsRequest
  object.

  Fields:
    execution: Required. The resource name of the Execution that the Events
      connect Artifacts with. Format: `projects/{project}/locations/{location}
      /metadataStores/{metadatastore}/executions/{execution}`
    googleCloudAiplatformV1AddExecutionEventsRequest: A
      GoogleCloudAiplatformV1AddExecutionEventsRequest resource to be passed
      as the request body.
  """
    execution = _messages.StringField(1, required=True)
    googleCloudAiplatformV1AddExecutionEventsRequest = _messages.MessageField('GoogleCloudAiplatformV1AddExecutionEventsRequest', 2)