from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegexValidation(_messages.Message):
    """Validation based on regular expressions.

  Fields:
    regexes: Required. RE2 regular expressions used to validate the
      parameter's value. The value must match the regex in its entirety
      (substring matches are not sufficient).
  """
    regexes = _messages.StringField(1, repeated=True)