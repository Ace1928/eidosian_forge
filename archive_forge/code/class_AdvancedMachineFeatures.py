from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AdvancedMachineFeatures(_messages.Message):
    """Specifies options for controlling advanced machine features.

  Fields:
    enableNestedVirtualization: Optional. Whether to enable nested
      virtualization or not (default is false).
    enableUefiNetworking: Optional. Whether to enable UEFI networking for
      instance creation.
    threadsPerCore: Optional. The number of threads per physical core. To
      disable simultaneous multithreading (SMT) set this to 1. If unset, the
      maximum number of threads supported per core by the underlying processor
      is assumed.
    visibleCoreCount: Optional. The number of physical cores to expose to an
      instance. Multiply by the number of threads per core to compute the
      total number of virtual CPUs to expose to the instance. If unset, the
      number of cores is inferred from the instance's nominal CPU count and
      the underlying platform's SMT width.
  """
    enableNestedVirtualization = _messages.BooleanField(1)
    enableUefiNetworking = _messages.BooleanField(2)
    threadsPerCore = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    visibleCoreCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)