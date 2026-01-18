from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimeAcceleratorConfig(_messages.Message):
    """Definition of the types of hardware accelerators that can be used. See
  [Compute Engine AcceleratorTypes](https://cloud.google.com/compute/docs/refe
  rence/beta/acceleratorTypes). Examples: * `nvidia-tesla-k80` * `nvidia-
  tesla-p100` * `nvidia-tesla-v100` * `nvidia-tesla-p4` * `nvidia-tesla-t4` *
  `nvidia-tesla-a100`

  Enums:
    TypeValueValuesEnum: Accelerator model.

  Fields:
    coreCount: Count of cores of this accelerator.
    type: Accelerator model.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Accelerator model.

    Values:
      ACCELERATOR_TYPE_UNSPECIFIED: Accelerator type is not specified.
      NVIDIA_TESLA_K80: Accelerator type is Nvidia Tesla K80.
      NVIDIA_TESLA_P100: Accelerator type is Nvidia Tesla P100.
      NVIDIA_TESLA_V100: Accelerator type is Nvidia Tesla V100.
      NVIDIA_TESLA_P4: Accelerator type is Nvidia Tesla P4.
      NVIDIA_TESLA_T4: Accelerator type is Nvidia Tesla T4.
      NVIDIA_TESLA_A100: Accelerator type is Nvidia Tesla A100 - 40GB.
      NVIDIA_L4: Accelerator type is Nvidia L4.
      TPU_V2: (Coming soon) Accelerator type is TPU V2.
      TPU_V3: (Coming soon) Accelerator type is TPU V3.
      NVIDIA_TESLA_T4_VWS: Accelerator type is NVIDIA Tesla T4 Virtual
        Workstations.
      NVIDIA_TESLA_P100_VWS: Accelerator type is NVIDIA Tesla P100 Virtual
        Workstations.
      NVIDIA_TESLA_P4_VWS: Accelerator type is NVIDIA Tesla P4 Virtual
        Workstations.
    """
        ACCELERATOR_TYPE_UNSPECIFIED = 0
        NVIDIA_TESLA_K80 = 1
        NVIDIA_TESLA_P100 = 2
        NVIDIA_TESLA_V100 = 3
        NVIDIA_TESLA_P4 = 4
        NVIDIA_TESLA_T4 = 5
        NVIDIA_TESLA_A100 = 6
        NVIDIA_L4 = 7
        TPU_V2 = 8
        TPU_V3 = 9
        NVIDIA_TESLA_T4_VWS = 10
        NVIDIA_TESLA_P100_VWS = 11
        NVIDIA_TESLA_P4_VWS = 12
    coreCount = _messages.IntegerField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)