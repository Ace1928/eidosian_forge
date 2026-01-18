from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourcePolicyGroupPlacementPolicy(_messages.Message):
    """A GroupPlacementPolicy specifies resource placement configuration. It
  specifies the failure bucket separation

  Enums:
    CollocationValueValuesEnum: Specifies network collocation

  Fields:
    availabilityDomainCount: The number of availability domains to spread
      instances across. If two instances are in different availability domain,
      they are not in the same low latency network.
    collocation: Specifies network collocation
    maxDistance: Specifies the number of max logical switches.
    sliceCount: Specifies the number of slices in a multislice workload.
    tpuTopology: Specifies the shape of the TPU slice
    vmCount: Number of VMs in this placement group. Google does not recommend
      that you use this field unless you use a compact policy and you want
      your policy to work only if it contains this exact number of VMs.
  """

    class CollocationValueValuesEnum(_messages.Enum):
        """Specifies network collocation

    Values:
      COLLOCATED: <no description>
      UNSPECIFIED_COLLOCATION: <no description>
    """
        COLLOCATED = 0
        UNSPECIFIED_COLLOCATION = 1
    availabilityDomainCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    collocation = _messages.EnumField('CollocationValueValuesEnum', 2)
    maxDistance = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    sliceCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    tpuTopology = _messages.StringField(5)
    vmCount = _messages.IntegerField(6, variant=_messages.Variant.INT32)