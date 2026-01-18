from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsAgentEntityTypesBatchDeleteRequest(_messages.Message):
    """A DialogflowProjectsAgentEntityTypesBatchDeleteRequest object.

  Fields:
    googleCloudDialogflowV2BatchDeleteEntityTypesRequest: A
      GoogleCloudDialogflowV2BatchDeleteEntityTypesRequest resource to be
      passed as the request body.
    parent: Required. The name of the agent to delete all entities types for.
      Format: `projects//agent`.
  """
    googleCloudDialogflowV2BatchDeleteEntityTypesRequest = _messages.MessageField('GoogleCloudDialogflowV2BatchDeleteEntityTypesRequest', 1)
    parent = _messages.StringField(2, required=True)