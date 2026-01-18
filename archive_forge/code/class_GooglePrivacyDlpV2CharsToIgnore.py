from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2CharsToIgnore(_messages.Message):
    """Characters to skip when doing deidentification of a value. These will be
  left alone and skipped.

  Enums:
    CommonCharactersToIgnoreValueValuesEnum: Common characters to not
      transform when masking. Useful to avoid removing punctuation.

  Fields:
    charactersToSkip: Characters to not transform when masking.
    commonCharactersToIgnore: Common characters to not transform when masking.
      Useful to avoid removing punctuation.
  """

    class CommonCharactersToIgnoreValueValuesEnum(_messages.Enum):
        """Common characters to not transform when masking. Useful to avoid
    removing punctuation.

    Values:
      COMMON_CHARS_TO_IGNORE_UNSPECIFIED: Unused.
      NUMERIC: 0-9
      ALPHA_UPPER_CASE: A-Z
      ALPHA_LOWER_CASE: a-z
      PUNCTUATION: US Punctuation, one of !"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~
      WHITESPACE: Whitespace character, one of [ \\t\\n\\x0B\\f\\r]
    """
        COMMON_CHARS_TO_IGNORE_UNSPECIFIED = 0
        NUMERIC = 1
        ALPHA_UPPER_CASE = 2
        ALPHA_LOWER_CASE = 3
        PUNCTUATION = 4
        WHITESPACE = 5
    charactersToSkip = _messages.StringField(1)
    commonCharactersToIgnore = _messages.EnumField('CommonCharactersToIgnoreValueValuesEnum', 2)