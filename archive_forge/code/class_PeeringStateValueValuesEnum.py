from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PeeringStateValueValuesEnum(_messages.Enum):
    """Output only. Peering state between service network and VMware Engine
    network.

    Values:
      PEERING_STATE_UNSPECIFIED: The default value. This value is used if the
        peering state is omitted or unknown.
      PEERING_ACTIVE: The peering is in active state.
      PEERING_INACTIVE: The peering is in inactive state.
    """
    PEERING_STATE_UNSPECIFIED = 0
    PEERING_ACTIVE = 1
    PEERING_INACTIVE = 2