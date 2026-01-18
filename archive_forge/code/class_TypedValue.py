from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TypedValue(_messages.Message):
    """A single strongly-typed value.

  Fields:
    boolValue: A Boolean value: true or false.
    distributionValue: A distribution value.
    doubleValue: A 64-bit double-precision floating-point number. Its
      magnitude is approximately \\xb110\\xb1300 and it has 16 significant
      digits of precision.
    int64Value: A 64-bit integer. Its range is approximately \\xb19.2x1018.
    stringValue: A variable-length string value.
  """
    boolValue = _messages.BooleanField(1)
    distributionValue = _messages.MessageField('Distribution', 2)
    doubleValue = _messages.FloatField(3)
    int64Value = _messages.IntegerField(4)
    stringValue = _messages.StringField(5)