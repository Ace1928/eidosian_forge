from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class FusiontablesStyleGetRequest(_messages.Message):
    """A FusiontablesStyleGetRequest object.

  Fields:
    styleId: Identifier (integer) for a specific style in a table
    tableId: Table to which the requested style belongs
  """
    styleId = _messages.IntegerField(1, required=True, variant=_messages.Variant.INT32)
    tableId = _messages.StringField(2, required=True)