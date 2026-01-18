from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OpValueValuesEnum(_messages.Enum):
    """An operator to apply the subject with.

    Values:
      NO_OP: Default no-op.
      EQUALS: DEPRECATED. Use IN instead.
      NOT_EQUALS: DEPRECATED. Use NOT_IN instead.
      IN: Set-inclusion check.
      NOT_IN: Set-exclusion check.
      DISCHARGED: Subject is discharged
    """
    NO_OP = 0
    EQUALS = 1
    NOT_EQUALS = 2
    IN = 3
    NOT_IN = 4
    DISCHARGED = 5