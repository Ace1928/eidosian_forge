from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourcePolicyVmMaintenancePolicy(_messages.Message):
    """A ResourcePolicyVmMaintenancePolicy object.

  Fields:
    concurrencyControlGroup: A
      ResourcePolicyVmMaintenancePolicyConcurrencyControl attribute.
    maintenanceWindow: Maintenance windows that are applied to VMs covered by
      this policy.
  """
    concurrencyControlGroup = _messages.MessageField('ResourcePolicyVmMaintenancePolicyConcurrencyControl', 1)
    maintenanceWindow = _messages.MessageField('ResourcePolicyVmMaintenancePolicyMaintenanceWindow', 2)