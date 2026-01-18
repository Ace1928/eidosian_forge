from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EvaluationMissingDataValueValuesEnum(_messages.Enum):
    """A condition control that determines how metric-threshold conditions
    are evaluated when data stops arriving.

    Values:
      EVALUATION_MISSING_DATA_UNSPECIFIED: An unspecified evaluation missing
        data option. Equivalent to EVALUATION_MISSING_DATA_NO_OP.
      EVALUATION_MISSING_DATA_INACTIVE: If there is no data to evaluate the
        condition, then evaluate the condition as false.
      EVALUATION_MISSING_DATA_ACTIVE: If there is no data to evaluate the
        condition, then evaluate the condition as true.
      EVALUATION_MISSING_DATA_NO_OP: Do not evaluate the condition to any
        value if there is no data.
    """
    EVALUATION_MISSING_DATA_UNSPECIFIED = 0
    EVALUATION_MISSING_DATA_INACTIVE = 1
    EVALUATION_MISSING_DATA_ACTIVE = 2
    EVALUATION_MISSING_DATA_NO_OP = 3