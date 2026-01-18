from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SuggestTrialsMetadata(_messages.Message):
    """Details of operations that perform Trials suggestion.

  Fields:
    clientId: The identifier of the client that is requesting the suggestion.
      If multiple SuggestTrialsRequests have the same `client_id`, the service
      will return the identical suggested Trial if the Trial is pending, and
      provide a new Trial if the last suggested Trial was completed.
    genericMetadata: Operation metadata for suggesting Trials.
  """
    clientId = _messages.StringField(1)
    genericMetadata = _messages.MessageField('GoogleCloudAiplatformV1GenericOperationMetadata', 2)