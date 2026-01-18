from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ConstraintDefaultValueValuesEnum(_messages.Enum):
    """The evaluation behavior of this constraint in the absence of a policy.

    Values:
      CONSTRAINT_DEFAULT_UNSPECIFIED: This is only used for distinguishing
        unset values and should never be used.
      ALLOW: Indicate that all values are allowed for list constraints.
        Indicate that enforcement is off for boolean constraints.
      DENY: Indicate that all values are denied for list constraints. Indicate
        that enforcement is on for boolean constraints.
    """
    CONSTRAINT_DEFAULT_UNSPECIFIED = 0
    ALLOW = 1
    DENY = 2