from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SidFilteringStateValueValuesEnum(_messages.Enum):
    """Current SID filtering state.

    Values:
      SID_FILTERING_STATE_UNSPECIFIED: SID Filtering is in unspecified state.
      ENABLED: SID Filtering is Enabled.
      DISABLED: SID Filtering is Disabled.
    """
    SID_FILTERING_STATE_UNSPECIFIED = 0
    ENABLED = 1
    DISABLED = 2