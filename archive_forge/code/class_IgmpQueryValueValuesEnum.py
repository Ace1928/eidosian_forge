from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IgmpQueryValueValuesEnum(_messages.Enum):
    """Indicate whether igmp query is enabled on the network interface or
    not. If enabled, also indicates the version of IGMP supported.

    Values:
      IGMP_QUERY_DISABLED: The network interface has disabled IGMP query.
      IGMP_QUERY_V2: The network interface has enabled IGMP query - v2.
    """
    IGMP_QUERY_DISABLED = 0
    IGMP_QUERY_V2 = 1