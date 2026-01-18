from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceConsumptionInfo(_messages.Message):
    """A InstanceConsumptionInfo object.

  Fields:
    guestCpus: The number of virtual CPUs that are available to the instance.
    localSsdGb: The amount of local SSD storage available to the instance,
      defined in GiB.
    memoryMb: The amount of physical memory available to the instance, defined
      in MiB.
    minNodeCpus: The minimal guaranteed number of virtual CPUs that are
      reserved.
  """
    guestCpus = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    localSsdGb = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    memoryMb = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    minNodeCpus = _messages.IntegerField(4, variant=_messages.Variant.INT32)