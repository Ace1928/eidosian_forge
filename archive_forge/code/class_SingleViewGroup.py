from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SingleViewGroup(_messages.Message):
    """A widget that groups the other widgets by using a dropdown menu. All
  widgets that are within the area spanned by the grouping widget are
  considered member widgets.
  """