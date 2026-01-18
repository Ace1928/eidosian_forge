from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetSizeUnitValueValuesEnum(_messages.Enum):
    """The unit of measure for the target size.

    Values:
      INSTANCE: [Default] TargetSize is the target number of instances.
      VCPU: TargetSize is the target count of vCPUs of VMs.
    """
    INSTANCE = 0
    VCPU = 1