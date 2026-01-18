from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CollapsibleGroup(_messages.Message):
    """A widget that groups the other widgets. All widgets that are within the
  area spanned by the grouping widget are considered member widgets.

  Fields:
    collapsed: The collapsed state of the widget on first page load.
  """
    collapsed = _messages.BooleanField(1)