from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentMessageSelectItemInfo(_messages.Message):
    """Additional info about the select item for when it is triggered in a
  dialog.

  Fields:
    key: Required. A unique key that will be sent back to the agent if this
      response is given.
    synonyms: Optional. A list of synonyms that can also be used to trigger
      this item in dialog.
  """
    key = _messages.StringField(1)
    synonyms = _messages.StringField(2, repeated=True)