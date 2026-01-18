from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentMessageBasicCardButton(_messages.Message):
    """The button object that appears at the bottom of a card.

  Fields:
    openUriAction: Required. Action to take when a user taps on the button.
    title: Required. The title of the button.
  """
    openUriAction = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageBasicCardButtonOpenUriAction', 1)
    title = _messages.StringField(2)