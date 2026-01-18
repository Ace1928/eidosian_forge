from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BgpRoutingModeValueValuesEnum(_messages.Enum):
    """Dynamic routing mode of the VPC network, `regional` or `global`.

    Values:
      BGP_ROUTING_MODE_UNSPECIFIED: Unknown.
      REGIONAL: Regional mode.
      GLOBAL: Global mode.
    """
    BGP_ROUTING_MODE_UNSPECIFIED = 0
    REGIONAL = 1
    GLOBAL = 2