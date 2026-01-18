from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsConversationsGetRequest(_messages.Message):
    """A DialogflowProjectsConversationsGetRequest object.

  Fields:
    name: Required. The name of the conversation. Format:
      `projects//locations//conversations/`.
  """
    name = _messages.StringField(1, required=True)