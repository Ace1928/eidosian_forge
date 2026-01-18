from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentMessageText(_messages.Message):
    """The text response message.

  Fields:
    text: Optional. The collection of the agent's responses.
  """
    text = _messages.StringField(1, repeated=True)