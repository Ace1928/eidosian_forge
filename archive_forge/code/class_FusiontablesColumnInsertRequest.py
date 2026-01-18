from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class FusiontablesColumnInsertRequest(_messages.Message):
    """A FusiontablesColumnInsertRequest object.

  Fields:
    column: A Column resource to be passed as the request body.
    tableId: Table for which a new column is being added.
  """
    column = _messages.MessageField('Column', 1)
    tableId = _messages.StringField(2, required=True)