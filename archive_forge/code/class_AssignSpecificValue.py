from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AssignSpecificValue(_messages.Message):
    """Set to a specific value (value is converted to fit the target data type)

  Fields:
    value: Required. Specific value to be assigned
  """
    value = _messages.StringField(1)