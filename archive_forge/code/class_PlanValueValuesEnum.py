from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PlanValueValuesEnum(_messages.Enum):
    """The plan for this commitment, which determines duration and discount
    rate. The currently supported plans are TWELVE_MONTH (1 year), and
    THIRTY_SIX_MONTH (3 years).

    Values:
      INVALID: <no description>
      THIRTY_SIX_MONTH: <no description>
      TWELVE_MONTH: <no description>
    """
    INVALID = 0
    THIRTY_SIX_MONTH = 1
    TWELVE_MONTH = 2