from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OutcomeValueValuesEnum(_messages.Enum):
    """Outcome of the configuration step.

    Values:
      OUTCOME_UNSPECIFIED: Default value. This value is unused.
      SUCCEEDED: The step succeeded.
      FAILED: The step failed.
    """
    OUTCOME_UNSPECIFIED = 0
    SUCCEEDED = 1
    FAILED = 2