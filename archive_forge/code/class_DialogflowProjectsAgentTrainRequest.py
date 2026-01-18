from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsAgentTrainRequest(_messages.Message):
    """A DialogflowProjectsAgentTrainRequest object.

  Fields:
    googleCloudDialogflowV2TrainAgentRequest: A
      GoogleCloudDialogflowV2TrainAgentRequest resource to be passed as the
      request body.
    parent: Required. The project that the agent to train is associated with.
      Format: `projects/`.
  """
    googleCloudDialogflowV2TrainAgentRequest = _messages.MessageField('GoogleCloudDialogflowV2TrainAgentRequest', 1)
    parent = _messages.StringField(2, required=True)