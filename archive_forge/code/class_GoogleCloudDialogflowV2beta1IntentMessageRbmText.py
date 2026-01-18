from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageRbmText(_messages.Message):
    """Rich Business Messaging (RBM) text response with suggestions.

  Fields:
    rbmSuggestion: Optional. One or more suggestions to show to the user.
    text: Required. Text sent and displayed to the user.
  """
    rbmSuggestion = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageRbmSuggestion', 1, repeated=True)
    text = _messages.StringField(2)