import logging
from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGP_PROCESSOR_ERROR_CODE
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.utils import circlist
from os_ken.services.protocols.bgp.utils.evtlet import EventletIOFactory
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_LOCAL_PREF
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGINATOR_ID
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_CLUSTER_LIST
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_IGP
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_EGP
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_INCOMPLETE
from os_ken.services.protocols.bgp.constants import VRF_TABLE
def _cmp_by_reachable_nh(path1, path2):
    """Compares given paths and selects best path based on reachable next-hop.

    If no path matches this criteria, return None.
    """
    return None