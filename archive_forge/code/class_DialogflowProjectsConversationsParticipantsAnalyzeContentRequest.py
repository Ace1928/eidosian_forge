from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsConversationsParticipantsAnalyzeContentRequest(_messages.Message):
    """A DialogflowProjectsConversationsParticipantsAnalyzeContentRequest
  object.

  Fields:
    googleCloudDialogflowV2AnalyzeContentRequest: A
      GoogleCloudDialogflowV2AnalyzeContentRequest resource to be passed as
      the request body.
    participant: Required. The name of the participant this text comes from.
      Format: `projects//locations//conversations//participants/`.
  """
    googleCloudDialogflowV2AnalyzeContentRequest = _messages.MessageField('GoogleCloudDialogflowV2AnalyzeContentRequest', 1)
    participant = _messages.StringField(2, required=True)