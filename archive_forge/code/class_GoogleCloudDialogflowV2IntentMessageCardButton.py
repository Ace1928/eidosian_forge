from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentMessageCardButton(_messages.Message):
    """Contains information about a button.

  Fields:
    postback: Optional. The text to send back to the Dialogflow API or a URI
      to open.
    text: Optional. The text to show on the button.
  """
    postback = _messages.StringField(1)
    text = _messages.StringField(2)