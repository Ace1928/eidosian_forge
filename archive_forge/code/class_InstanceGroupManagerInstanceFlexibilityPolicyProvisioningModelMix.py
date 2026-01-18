from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagerInstanceFlexibilityPolicyProvisioningModelMix(_messages.Message):
    """A InstanceGroupManagerInstanceFlexibilityPolicyProvisioningModelMix
  object.

  Fields:
    standardCapacityBase: The base capacity that will always use Standard VMs
      to avoid risk of more preemption than the minimum capacity user needs.
      MIG will create only Standard VMs until it reaches
      standard_capacity_base and only then will start using
      standard_capacity_percent_above_base to mix Spot with Standard VMs.
    standardCapacityPercentAboveBase: The percentage of target capacity that
      should use Standard VM. The remaining percentage will use Spot VMs. The
      percentage applies only to the capacity above standard_capacity_base.
  """
    standardCapacityBase = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    standardCapacityPercentAboveBase = _messages.IntegerField(2, variant=_messages.Variant.INT32)