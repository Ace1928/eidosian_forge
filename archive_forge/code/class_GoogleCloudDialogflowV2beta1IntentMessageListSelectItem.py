from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageListSelectItem(_messages.Message):
    """An item in the list.

  Fields:
    description: Optional. The main text describing the item.
    image: Optional. The image to display.
    info: Required. Additional information about this option.
    title: Required. The title of the list item.
  """
    description = _messages.StringField(1)
    image = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageImage', 2)
    info = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageSelectItemInfo', 3)
    title = _messages.StringField(4)