from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsConversationProfilesDeleteRequest(_messages.Message):
    """A DialogflowProjectsConversationProfilesDeleteRequest object.

  Fields:
    name: Required. The name of the conversation profile to delete. Format:
      `projects//locations//conversationProfiles/`.
  """
    name = _messages.StringField(1, required=True)