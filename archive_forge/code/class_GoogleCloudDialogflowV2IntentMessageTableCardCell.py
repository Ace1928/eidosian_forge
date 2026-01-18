from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentMessageTableCardCell(_messages.Message):
    """Cell of TableCardRow.

  Fields:
    text: Required. Text in this cell.
  """
    text = _messages.StringField(1)