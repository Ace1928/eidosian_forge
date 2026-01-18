from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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