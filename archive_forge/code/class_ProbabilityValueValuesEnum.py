from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProbabilityValueValuesEnum(_messages.Enum):
    """Output only. Harm probability levels in the content.

    Values:
      HARM_PROBABILITY_UNSPECIFIED: Harm probability unspecified.
      NEGLIGIBLE: Negligible level of harm.
      LOW: Low level of harm.
      MEDIUM: Medium level of harm.
      HIGH: High level of harm.
    """
    HARM_PROBABILITY_UNSPECIFIED = 0
    NEGLIGIBLE = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4