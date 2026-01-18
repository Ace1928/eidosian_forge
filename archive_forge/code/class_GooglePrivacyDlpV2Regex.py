from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2Regex(_messages.Message):
    """Message defining a custom regular expression.

  Fields:
    groupIndexes: The index of the submatch to extract as findings. When not
      specified, the entire match is returned. No more than 3 may be included.
    pattern: Pattern defining the regular expression. Its syntax
      (https://github.com/google/re2/wiki/Syntax) can be found under the
      google/re2 repository on GitHub.
  """
    groupIndexes = _messages.IntegerField(1, repeated=True, variant=_messages.Variant.INT32)
    pattern = _messages.StringField(2)