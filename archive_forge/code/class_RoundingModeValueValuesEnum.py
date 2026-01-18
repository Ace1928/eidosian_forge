from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RoundingModeValueValuesEnum(_messages.Enum):
    """Optional. Specifies the rounding mode to be used when storing values
    of NUMERIC and BIGNUMERIC type.

    Values:
      ROUNDING_MODE_UNSPECIFIED: Unspecified will default to using
        ROUND_HALF_AWAY_FROM_ZERO.
      ROUND_HALF_AWAY_FROM_ZERO: ROUND_HALF_AWAY_FROM_ZERO rounds half values
        away from zero when applying precision and scale upon writing of
        NUMERIC and BIGNUMERIC values. For Scale: 0 1.1, 1.2, 1.3, 1.4 => 1
        1.5, 1.6, 1.7, 1.8, 1.9 => 2
      ROUND_HALF_EVEN: ROUND_HALF_EVEN rounds half values to the nearest even
        value when applying precision and scale upon writing of NUMERIC and
        BIGNUMERIC values. For Scale: 0 1.1, 1.2, 1.3, 1.4 => 1 1.5 => 2 1.6,
        1.7, 1.8, 1.9 => 2 2.5 => 2
    """
    ROUNDING_MODE_UNSPECIFIED = 0
    ROUND_HALF_AWAY_FROM_ZERO = 1
    ROUND_HALF_EVEN = 2