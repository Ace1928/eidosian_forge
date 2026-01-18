from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IntegerList(_messages.Message):
    """A metric value representing a list of integers.

  Fields:
    elements: Elements of the list.
  """
    elements = _messages.MessageField('SplitInt64', 1, repeated=True)