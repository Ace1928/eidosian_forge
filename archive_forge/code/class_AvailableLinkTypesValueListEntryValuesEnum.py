from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AvailableLinkTypesValueListEntryValuesEnum(_messages.Enum):
    """AvailableLinkTypesValueListEntryValuesEnum enum type.

    Values:
      LINK_TYPE_ETHERNET_100G_LR: 100G Ethernet, LR Optics.
      LINK_TYPE_ETHERNET_10G_LR: 10G Ethernet, LR Optics. [(rate_bps) =
        10000000000];
    """
    LINK_TYPE_ETHERNET_100G_LR = 0
    LINK_TYPE_ETHERNET_10G_LR = 1