from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2SuggestArticlesRequest(_messages.Message):
    """The request message for Participants.SuggestArticles.

  Fields:
    assistQueryParams: Parameters for a human assist query.
    contextSize: Optional. Max number of messages prior to and including
      latest_message to use as context when compiling the suggestion. By
      default 20 and at most 50.
    latestMessage: Optional. The name of the latest conversation message to
      compile suggestion for. If empty, it will be the latest message of the
      conversation. Format: `projects//locations//conversations//messages/`.
  """
    assistQueryParams = _messages.MessageField('GoogleCloudDialogflowV2AssistQueryParameters', 1)
    contextSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    latestMessage = _messages.StringField(3)