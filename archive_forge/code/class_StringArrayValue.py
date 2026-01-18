from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StringArrayValue(_messages.Message):
    """An array of strings within a parameter.

  Fields:
    values: Required. The values of the array.
  """
    values = _messages.StringField(1, repeated=True)