from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TokenizationValueValuesEnum(_messages.Enum):
    """String tokenization mode.

    Values:
      TOKENIZATION_UNSPECIFIED: No tokenization - only exact string matches
        are supported.
      WORDS: Use word tokens.
      SUBSTRINGS_NGRAM_3: Uses 3-ngram tokens supporting efficient substring
        searches.
    """
    TOKENIZATION_UNSPECIFIED = 0
    WORDS = 1
    SUBSTRINGS_NGRAM_3 = 2