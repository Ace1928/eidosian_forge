from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CommonAlphabetValueValuesEnum(_messages.Enum):
    """Common alphabets.

    Values:
      FFX_COMMON_NATIVE_ALPHABET_UNSPECIFIED: Unused.
      NUMERIC: `[0-9]` (radix of 10)
      HEXADECIMAL: `[0-9A-F]` (radix of 16)
      UPPER_CASE_ALPHA_NUMERIC: `[0-9A-Z]` (radix of 36)
      ALPHA_NUMERIC: `[0-9A-Za-z]` (radix of 62)
    """
    FFX_COMMON_NATIVE_ALPHABET_UNSPECIFIED = 0
    NUMERIC = 1
    HEXADECIMAL = 2
    UPPER_CASE_ALPHA_NUMERIC = 3
    ALPHA_NUMERIC = 4