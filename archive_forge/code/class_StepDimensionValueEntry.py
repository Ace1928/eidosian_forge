from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StepDimensionValueEntry(_messages.Message):
    """A StepDimensionValueEntry object.

  Fields:
    key: A string attribute.
    value: A string attribute.
  """
    key = _messages.StringField(1)
    value = _messages.StringField(2)