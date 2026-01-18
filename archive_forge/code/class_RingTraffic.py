from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RingTraffic(_messages.Message):
    """Predefined traffic shape in which each (i_th) `group` member sends
  traffic to the subsequent (i_th+1) `group` member, looping around at the
  end.

  Enums:
    TrafficDirectionValueValuesEnum: If UNSPECIFIED or UNIDIRECTIONAL, the
      i_th member sends traffic to the i+1_th member, looping at the end. If
      BIDIRECTIONAL, the i_th member additionally sends traffic to the i-1_th
      member, looping at the start.

  Fields:
    group: Sorted list of coordinates participating in the Ring Traffic
      exchange.
    trafficDirection: If UNSPECIFIED or UNIDIRECTIONAL, the i_th member sends
      traffic to the i+1_th member, looping at the end. If BIDIRECTIONAL, the
      i_th member additionally sends traffic to the i-1_th member, looping at
      the start.
  """

    class TrafficDirectionValueValuesEnum(_messages.Enum):
        """If UNSPECIFIED or UNIDIRECTIONAL, the i_th member sends traffic to the
    i+1_th member, looping at the end. If BIDIRECTIONAL, the i_th member
    additionally sends traffic to the i-1_th member, looping at the start.

    Values:
      TRAFFIC_DIRECTION_UNSPECIFIED: Traffic direction is not specified
      TRAFFIC_DIRECTION_UNIDIRECTIONAL: Traffic is sent in one direction.
      TRAFFIC_DIRECTION_BIDIRECTIONAL: Traffic is sent in both directions.
    """
        TRAFFIC_DIRECTION_UNSPECIFIED = 0
        TRAFFIC_DIRECTION_UNIDIRECTIONAL = 1
        TRAFFIC_DIRECTION_BIDIRECTIONAL = 2
    group = _messages.MessageField('CoordinateList', 1)
    trafficDirection = _messages.EnumField('TrafficDirectionValueValuesEnum', 2)