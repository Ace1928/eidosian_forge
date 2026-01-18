from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsAgentEntityTypesEntitiesBatchDeleteRequest(_messages.Message):
    """A DialogflowProjectsAgentEntityTypesEntitiesBatchDeleteRequest object.

  Fields:
    googleCloudDialogflowV2BatchDeleteEntitiesRequest: A
      GoogleCloudDialogflowV2BatchDeleteEntitiesRequest resource to be passed
      as the request body.
    parent: Required. The name of the entity type to delete entries for.
      Format: `projects//agent/entityTypes/`.
  """
    googleCloudDialogflowV2BatchDeleteEntitiesRequest = _messages.MessageField('GoogleCloudDialogflowV2BatchDeleteEntitiesRequest', 1)
    parent = _messages.StringField(2, required=True)