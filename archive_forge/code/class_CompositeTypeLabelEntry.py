from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CompositeTypeLabelEntry(_messages.Message):
    """Label object for CompositeTypes

  Fields:
    key: Key of the label
    value: Value of the label
  """
    key = _messages.StringField(1)
    value = _messages.StringField(2)