from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class FusiontablesTableGetRequest(_messages.Message):
    """A FusiontablesTableGetRequest object.

  Fields:
    tableId: Identifier(ID) for the table being requested.
  """
    tableId = _messages.StringField(1, required=True)