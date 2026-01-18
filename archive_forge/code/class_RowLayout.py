from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RowLayout(_messages.Message):
    """A simplified layout that divides the available space into rows and
  arranges a set of widgets horizontally in each row.

  Fields:
    rows: The rows of content to display.
  """
    rows = _messages.MessageField('Row', 1, repeated=True)