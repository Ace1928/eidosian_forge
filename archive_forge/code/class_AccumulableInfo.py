from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccumulableInfo(_messages.Message):
    """A AccumulableInfo object.

  Fields:
    accumullableInfoId: A string attribute.
    name: A string attribute.
    update: A string attribute.
    value: A string attribute.
  """
    accumullableInfoId = _messages.IntegerField(1)
    name = _messages.StringField(2)
    update = _messages.StringField(3)
    value = _messages.StringField(4)