from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SchedulerAcceleratorConfig(_messages.Message):
    """Definition of a hardware accelerator. Note that not all combinations of
  `type` and `core_count` are valid. See [GPUs on Compute
  Engine](https://cloud.google.com/compute/docs/gpus) to find a valid
  combination. TPUs are not supported.

  Enums:
    TypeValueValuesEnum: Type of this accelerator.

  Fields:
    coreCount: Count of cores of this accelerator.
    type: Type of this accelerator.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Type of this accelerator.

    Values:
      SCHEDULER_ACCELERATOR_TYPE_UNSPECIFIED: Unspecified accelerator type.
        Default to no GPU.
      NVIDIA_TESLA_K80: Nvidia Tesla K80 GPU.
      NVIDIA_TESLA_P100: Nvidia Tesla P100 GPU.
      NVIDIA_TESLA_V100: Nvidia Tesla V100 GPU.
      NVIDIA_TESLA_P4: Nvidia Tesla P4 GPU.
      NVIDIA_TESLA_T4: Nvidia Tesla T4 GPU.
      NVIDIA_TESLA_A100: Nvidia Tesla A100 GPU.
      TPU_V2: TPU v2.
      TPU_V3: TPU v3.
    """
        SCHEDULER_ACCELERATOR_TYPE_UNSPECIFIED = 0
        NVIDIA_TESLA_K80 = 1
        NVIDIA_TESLA_P100 = 2
        NVIDIA_TESLA_V100 = 3
        NVIDIA_TESLA_P4 = 4
        NVIDIA_TESLA_T4 = 5
        NVIDIA_TESLA_A100 = 6
        TPU_V2 = 7
        TPU_V3 = 8
    coreCount = _messages.IntegerField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)