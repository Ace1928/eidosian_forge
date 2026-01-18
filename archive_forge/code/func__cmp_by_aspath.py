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
def _cmp_by_aspath(path1, path2):
    """Calculated the best-paths by comparing as-path lengths.

    Shortest as-path length is preferred. If both path have same lengths,
    we return None.
    """
    as_path1 = path1.get_pattr(BGP_ATTR_TYPE_AS_PATH)
    as_path2 = path2.get_pattr(BGP_ATTR_TYPE_AS_PATH)
    assert as_path1 and as_path2
    l1 = as_path1.get_as_path_len()
    l2 = as_path2.get_as_path_len()
    assert l1 is not None and l2 is not None
    if l1 > l2:
        return path2
    elif l2 > l1:
        return path1
    else:
        return None