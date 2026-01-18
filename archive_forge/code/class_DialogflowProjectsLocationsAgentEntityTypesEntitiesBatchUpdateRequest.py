from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAgentEntityTypesEntitiesBatchUpdateRequest(_messages.Message):
    """A DialogflowProjectsLocationsAgentEntityTypesEntitiesBatchUpdateRequest
  object.

  Fields:
    googleCloudDialogflowV2BatchUpdateEntitiesRequest: A
      GoogleCloudDialogflowV2BatchUpdateEntitiesRequest resource to be passed
      as the request body.
    parent: Required. The name of the entity type to update or create entities
      in. Format: `projects//agent/entityTypes/`.
  """
    googleCloudDialogflowV2BatchUpdateEntitiesRequest = _messages.MessageField('GoogleCloudDialogflowV2BatchUpdateEntitiesRequest', 1)
    parent = _messages.StringField(2, required=True)