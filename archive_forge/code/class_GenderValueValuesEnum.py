from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GenderValueValuesEnum(_messages.Enum):
    """The grammatical gender.

    Values:
      GENDER_UNKNOWN: Gender is not applicable in the analyzed language or is
        not predicted.
      FEMININE: Feminine
      MASCULINE: Masculine
      NEUTER: Neuter
    """
    GENDER_UNKNOWN = 0
    FEMININE = 1
    MASCULINE = 2
    NEUTER = 3