from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class FusiontablesStyleDeleteRequest(_messages.Message):
    """A FusiontablesStyleDeleteRequest object.

  Fields:
    styleId: Identifier (within a table) for the style being deleted
    tableId: Table from which the style is being deleted
  """
    styleId = _messages.IntegerField(1, required=True, variant=_messages.Variant.INT32)
    tableId = _messages.StringField(2, required=True)