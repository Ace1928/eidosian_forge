from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAgentIntentsBatchDeleteRequest(_messages.Message):
    """A DialogflowProjectsLocationsAgentIntentsBatchDeleteRequest object.

  Fields:
    googleCloudDialogflowV2BatchDeleteIntentsRequest: A
      GoogleCloudDialogflowV2BatchDeleteIntentsRequest resource to be passed
      as the request body.
    parent: Required. The name of the agent to delete all entities types for.
      Format: `projects//agent`.
  """
    googleCloudDialogflowV2BatchDeleteIntentsRequest = _messages.MessageField('GoogleCloudDialogflowV2BatchDeleteIntentsRequest', 1)
    parent = _messages.StringField(2, required=True)