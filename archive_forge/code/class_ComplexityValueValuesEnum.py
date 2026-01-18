from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ComplexityValueValuesEnum(_messages.Enum):
    """The complexity of the password.

    Values:
      COMPLEXITY_UNSPECIFIED: Complexity check is not specified.
      COMPLEXITY_DEFAULT: A combination of lowercase, uppercase, numeric, and
        non-alphanumeric characters.
    """
    COMPLEXITY_UNSPECIFIED = 0
    COMPLEXITY_DEFAULT = 1