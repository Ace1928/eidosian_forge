from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OnPolicySwitchValueValuesEnum(_messages.Enum):
    """OnPolicySwitchValueValuesEnum enum type.

    Values:
      DO_NOT_RETROACTIVELY_APPLY: <no description>
      RETROACTIVELY_APPLY: <no description>
      UNSPECIFIED_ON_POLICY_SWITCH: <no description>
    """
    DO_NOT_RETROACTIVELY_APPLY = 0
    RETROACTIVELY_APPLY = 1
    UNSPECIFIED_ON_POLICY_SWITCH = 2