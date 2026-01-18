from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValueValidation(_messages.Message):
    """Validation based on a list of allowed values.

  Fields:
    values: Required. List of allowed values for the parameter.
  """
    values = _messages.StringField(1, repeated=True)