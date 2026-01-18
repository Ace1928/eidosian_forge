from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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