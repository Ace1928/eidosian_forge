from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ParameterValidation(_messages.Message):
    """Configuration for parameter validation.

  Fields:
    regex: Validation based on regular expressions.
    values: Validation based on a list of allowed values.
  """
    regex = _messages.MessageField('RegexValidation', 1)
    values = _messages.MessageField('ValueValidation', 2)