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
def _cmp_by_med(path1, path2):
    """Select the path based with lowest MED value.

    If both paths have same MED, return None.
    By default, a route that arrives with no MED value is treated as if it
    had a MED of 0, the most preferred value.
    RFC says lower MED is preferred over higher MED value.
    """

    def get_path_med(path):
        med = path.get_pattr(BGP_ATTR_TYPE_MULTI_EXIT_DISC)
        if not med:
            return 0
        return med.value
    med1 = get_path_med(path1)
    med2 = get_path_med(path2)
    if med1 == med2:
        return None
    elif med1 < med2:
        return path1
    return path2