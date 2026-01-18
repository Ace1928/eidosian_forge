from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceCommitment(_messages.Message):
    """Commitment for a particular resource (a Commitment is composed of one or
  more of these).

  Enums:
    TypeValueValuesEnum: Type of resource for which this commitment applies.
      Possible values are VCPU, MEMORY, LOCAL_SSD, and ACCELERATOR.

  Fields:
    acceleratorType: Name of the accelerator type resource. Applicable only
      when the type is ACCELERATOR.
    amount: The amount of the resource purchased (in a type-dependent unit,
      such as bytes). For vCPUs, this can just be an integer. For memory, this
      must be provided in MB. Memory must be a multiple of 256 MB, with up to
      6.5GB of memory per every vCPU.
    type: Type of resource for which this commitment applies. Possible values
      are VCPU, MEMORY, LOCAL_SSD, and ACCELERATOR.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Type of resource for which this commitment applies. Possible values
    are VCPU, MEMORY, LOCAL_SSD, and ACCELERATOR.

    Values:
      ACCELERATOR: <no description>
      LOCAL_SSD: <no description>
      MEMORY: <no description>
      UNSPECIFIED: <no description>
      VCPU: <no description>
    """
        ACCELERATOR = 0
        LOCAL_SSD = 1
        MEMORY = 2
        UNSPECIFIED = 3
        VCPU = 4
    acceleratorType = _messages.StringField(1)
    amount = _messages.IntegerField(2)
    type = _messages.EnumField('TypeValueValuesEnum', 3)