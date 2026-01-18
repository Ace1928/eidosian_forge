from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunOptionValueValuesEnum(_messages.Enum):
    """RunOptionValueValuesEnum enum type.

    Values:
      RUN_OPTION_UNSPECIFIED: Unspecified run option.
      AD_HOC: Ad-hoc run option.
      SCHEDULED: Scheduled run option.
    """
    RUN_OPTION_UNSPECIFIED = 0
    AD_HOC = 1
    SCHEDULED = 2