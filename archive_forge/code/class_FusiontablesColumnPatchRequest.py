from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class FusiontablesColumnPatchRequest(_messages.Message):
    """A FusiontablesColumnPatchRequest object.

  Fields:
    column: A Column resource to be passed as the request body.
    columnId: Name or identifier for the column that is being updated.
    tableId: Table for which the column is being updated.
  """
    column = _messages.MessageField('Column', 1)
    columnId = _messages.StringField(2, required=True)
    tableId = _messages.StringField(3, required=True)