from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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