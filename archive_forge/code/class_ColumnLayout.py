from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ColumnLayout(_messages.Message):
    """A simplified layout that divides the available space into vertical
  columns and arranges a set of widgets vertically in each column.

  Fields:
    columns: The columns of content to display.
  """
    columns = _messages.MessageField('Column', 1, repeated=True)