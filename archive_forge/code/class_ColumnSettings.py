from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ColumnSettings(_messages.Message):
    """The persistent settings for a table's columns.

  Fields:
    column: Required. The id of the column.
    visible: Required. Whether the column should be visible on page load.
  """
    column = _messages.StringField(1)
    visible = _messages.BooleanField(2)