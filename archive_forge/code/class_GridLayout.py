from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GridLayout(_messages.Message):
    """A basic layout divides the available space into vertical columns of
  equal width and arranges a list of widgets using a row-first strategy.

  Fields:
    columns: The number of columns into which the view's width is divided. If
      omitted or set to zero, a system default will be used while rendering.
    widgets: The informational elements that are arranged into the columns
      row-first.
  """
    columns = _messages.IntegerField(1)
    widgets = _messages.MessageField('Widget', 2, repeated=True)