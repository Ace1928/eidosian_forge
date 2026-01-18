from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BiEngineStatistics(_messages.Message):
    """Statistics for a BI Engine specific query. Populated as part of
  JobStatistics2

  Enums:
    AccelerationModeValueValuesEnum: Output only. Specifies which mode of BI
      Engine acceleration was performed (if any).
    BiEngineModeValueValuesEnum: Output only. Specifies which mode of BI
      Engine acceleration was performed (if any).

  Fields:
    accelerationMode: Output only. Specifies which mode of BI Engine
      acceleration was performed (if any).
    biEngineMode: Output only. Specifies which mode of BI Engine acceleration
      was performed (if any).
    biEngineReasons: In case of DISABLED or PARTIAL bi_engine_mode, these
      contain the explanatory reasons as to why BI Engine could not
      accelerate. In case the full query was accelerated, this field is not
      populated.
  """

    class AccelerationModeValueValuesEnum(_messages.Enum):
        """Output only. Specifies which mode of BI Engine acceleration was
    performed (if any).

    Values:
      BI_ENGINE_ACCELERATION_MODE_UNSPECIFIED: BiEngineMode type not
        specified.
      BI_ENGINE_DISABLED: BI Engine acceleration was attempted but disabled.
        bi_engine_reasons specifies a more detailed reason.
      PARTIAL_INPUT: Some inputs were accelerated using BI Engine. See
        bi_engine_reasons for why parts of the query were not accelerated.
      FULL_INPUT: All of the query inputs were accelerated using BI Engine.
      FULL_QUERY: All of the query was accelerated using BI Engine.
    """
        BI_ENGINE_ACCELERATION_MODE_UNSPECIFIED = 0
        BI_ENGINE_DISABLED = 1
        PARTIAL_INPUT = 2
        FULL_INPUT = 3
        FULL_QUERY = 4

    class BiEngineModeValueValuesEnum(_messages.Enum):
        """Output only. Specifies which mode of BI Engine acceleration was
    performed (if any).

    Values:
      ACCELERATION_MODE_UNSPECIFIED: BiEngineMode type not specified.
      DISABLED: BI Engine disabled the acceleration. bi_engine_reasons
        specifies a more detailed reason.
      PARTIAL: Part of the query was accelerated using BI Engine. See
        bi_engine_reasons for why parts of the query were not accelerated.
      FULL: All of the query was accelerated using BI Engine.
    """
        ACCELERATION_MODE_UNSPECIFIED = 0
        DISABLED = 1
        PARTIAL = 2
        FULL = 3
    accelerationMode = _messages.EnumField('AccelerationModeValueValuesEnum', 1)
    biEngineMode = _messages.EnumField('BiEngineModeValueValuesEnum', 2)
    biEngineReasons = _messages.MessageField('BiEngineReason', 3, repeated=True)