from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ForceUpdateOnRepairValueValuesEnum(_messages.Enum):
    """A bit indicating whether to forcefully apply the group's latest
    configuration when repairing a VM. Valid options are: - NO (default): If
    configuration updates are available, they are not forcefully applied
    during repair. Instead, configuration updates are applied according to the
    group's update policy. - YES: If configuration updates are available, they
    are applied during repair.

    Values:
      NO: <no description>
      YES: <no description>
    """
    NO = 0
    YES = 1