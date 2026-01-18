from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LogicalOperatorValueValuesEnum(_messages.Enum):
    """The logical operator to use between the fields and conditions.

    Values:
      OPERATOR_UNSPECIFIED: The default value.
      AND: AND operator; The conditions must all be true.
      OR: OR operator; At least one of the conditions must be true.
    """
    OPERATOR_UNSPECIFIED = 0
    AND = 1
    OR = 2