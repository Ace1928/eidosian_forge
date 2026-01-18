from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProvisioningModelMix(_messages.Message):
    """Defines how Dataproc should create VMs with a mixture of provisioning
  models.

  Fields:
    standardCapacityBase: Optional. The base capacity that will always use
      Standard VMs to avoid risk of more preemption than the minimum capacity
      you need. Dataproc will create only standard VMs until it reaches
      standardCapacityBaseNumber, then it will starting
      standardCapacityPercentAboveBase to mix Spot with Standard VMs.
    standardCapacityPercentAboveBase: Optional. The percentage of the capacity
      above standardCapacityBase that should use Spot VMs. The remaining
      percentage will use Standard VMs.
  """
    standardCapacityBase = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    standardCapacityPercentAboveBase = _messages.IntegerField(2, variant=_messages.Variant.INT32)