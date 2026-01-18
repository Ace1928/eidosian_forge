from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsConversationModelsDeployRequest(_messages.Message):
    """A DialogflowProjectsConversationModelsDeployRequest object.

  Fields:
    googleCloudDialogflowV2DeployConversationModelRequest: A
      GoogleCloudDialogflowV2DeployConversationModelRequest resource to be
      passed as the request body.
    name: Required. The conversation model to deploy. Format:
      `projects//conversationModels/`
  """
    googleCloudDialogflowV2DeployConversationModelRequest = _messages.MessageField('GoogleCloudDialogflowV2DeployConversationModelRequest', 1)
    name = _messages.StringField(2, required=True)