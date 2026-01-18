from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FaultTypeValueValuesEnum(_messages.Enum):
    """Required. The type of fault to be injected in an instance.

    Values:
      FAULT_TYPE_UNSPECIFIED: The fault type is unknown.
      STOP_VM: Stop the VM
    """
    FAULT_TYPE_UNSPECIFIED = 0
    STOP_VM = 1