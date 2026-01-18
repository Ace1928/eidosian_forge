from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourcePolicyVmMaintenancePolicyMaintenanceWindow(_messages.Message):
    """A maintenance window for VMs. When set, we restrict our maintenance
  operations to this window.

  Fields:
    dailyMaintenanceWindow: A ResourcePolicyDailyCycle attribute.
  """
    dailyMaintenanceWindow = _messages.MessageField('ResourcePolicyDailyCycle', 1)