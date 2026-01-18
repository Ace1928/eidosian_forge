from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CorrectnessLevelValueValuesEnum(_messages.Enum):
    """The correctness level of the specific answer.

    Values:
      CORRECTNESS_LEVEL_UNSPECIFIED: Correctness level unspecified.
      NOT_CORRECT: Answer is totally wrong.
      PARTIALLY_CORRECT: Answer is partially correct.
      FULLY_CORRECT: Answer is fully correct.
    """
    CORRECTNESS_LEVEL_UNSPECIFIED = 0
    NOT_CORRECT = 1
    PARTIALLY_CORRECT = 2
    FULLY_CORRECT = 3