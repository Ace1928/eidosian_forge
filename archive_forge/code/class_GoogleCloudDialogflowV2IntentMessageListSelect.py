from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentMessageListSelect(_messages.Message):
    """The card for presenting a list of options to select from.

  Fields:
    items: Required. List items.
    subtitle: Optional. Subtitle of the list.
    title: Optional. The overall title of the list.
  """
    items = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageListSelectItem', 1, repeated=True)
    subtitle = _messages.StringField(2)
    title = _messages.StringField(3)