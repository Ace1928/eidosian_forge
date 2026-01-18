from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LinkTypeValueValuesEnum(_messages.Enum):
    """Type of link requested, which can take one of the following values: -
    LINK_TYPE_ETHERNET_10G_LR: A 10G Ethernet with LR optics -
    LINK_TYPE_ETHERNET_100G_LR: A 100G Ethernet with LR optics. Note that this
    field indicates the speed of each of the links in the bundle, not the
    speed of the entire bundle.

    Values:
      LINK_TYPE_ETHERNET_100G_LR: 100G Ethernet, LR Optics.
      LINK_TYPE_ETHERNET_10G_LR: 10G Ethernet, LR Optics. [(rate_bps) =
        10000000000];
    """
    LINK_TYPE_ETHERNET_100G_LR = 0
    LINK_TYPE_ETHERNET_10G_LR = 1