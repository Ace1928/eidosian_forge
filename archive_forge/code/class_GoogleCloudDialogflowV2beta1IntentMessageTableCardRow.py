from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageTableCardRow(_messages.Message):
    """Row of TableCard.

  Fields:
    cells: Optional. List of cells that make up this row.
    dividerAfter: Optional. Whether to add a visual divider after this row.
  """
    cells = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageTableCardCell', 1, repeated=True)
    dividerAfter = _messages.BooleanField(2)