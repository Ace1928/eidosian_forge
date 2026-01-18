from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1Capability(_messages.Message):
    """A GoogleCloudMlV1Capability object.

  Enums:
    AvailableAcceleratorsValueListEntryValuesEnum:
    TypeValueValuesEnum:

  Fields:
    availableAccelerators: Available accelerators for the capability.
    type: A TypeValueValuesEnum attribute.
  """

    class AvailableAcceleratorsValueListEntryValuesEnum(_messages.Enum):
        """AvailableAcceleratorsValueListEntryValuesEnum enum type.

    Values:
      ACCELERATOR_TYPE_UNSPECIFIED: Unspecified accelerator type. Default to
        no GPU.
      NVIDIA_TESLA_K80: Nvidia Tesla K80 GPU.
      NVIDIA_TESLA_P100: Nvidia Tesla P100 GPU.
      NVIDIA_TESLA_V100: Nvidia V100 GPU.
      NVIDIA_TESLA_P4: Nvidia Tesla P4 GPU.
      NVIDIA_TESLA_T4: Nvidia T4 GPU.
      NVIDIA_TESLA_A100: Nvidia A100 GPU.
      TPU_V2: TPU v2.
      TPU_V3: TPU v3.
      TPU_V2_POD: TPU v2 POD.
      TPU_V3_POD: TPU v3 POD.
      TPU_V4_POD: TPU v4 POD.
    """
        ACCELERATOR_TYPE_UNSPECIFIED = 0
        NVIDIA_TESLA_K80 = 1
        NVIDIA_TESLA_P100 = 2
        NVIDIA_TESLA_V100 = 3
        NVIDIA_TESLA_P4 = 4
        NVIDIA_TESLA_T4 = 5
        NVIDIA_TESLA_A100 = 6
        TPU_V2 = 7
        TPU_V3 = 8
        TPU_V2_POD = 9
        TPU_V3_POD = 10
        TPU_V4_POD = 11

    class TypeValueValuesEnum(_messages.Enum):
        """TypeValueValuesEnum enum type.

    Values:
      TYPE_UNSPECIFIED: <no description>
      TRAINING: <no description>
      BATCH_PREDICTION: <no description>
      ONLINE_PREDICTION: <no description>
    """
        TYPE_UNSPECIFIED = 0
        TRAINING = 1
        BATCH_PREDICTION = 2
        ONLINE_PREDICTION = 3
    availableAccelerators = _messages.EnumField('AvailableAcceleratorsValueListEntryValuesEnum', 1, repeated=True)
    type = _messages.EnumField('TypeValueValuesEnum', 2)