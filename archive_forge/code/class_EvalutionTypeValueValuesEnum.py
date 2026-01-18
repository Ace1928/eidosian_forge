from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EvalutionTypeValueValuesEnum(_messages.Enum):
    """The evaluation type of the data quality rule.

    Values:
      EVALUATION_TYPE_UNSPECIFIED: An unspecified evaluation type.
      PER_ROW: The rule evaluation is done at per row level.
      AGGREGATE: The rule evaluation is done for an aggregate of rows.
    """
    EVALUATION_TYPE_UNSPECIFIED = 0
    PER_ROW = 1
    AGGREGATE = 2