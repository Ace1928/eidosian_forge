from os_ken.services.protocols.bgp.operator.views.base import \
from os_ken.services.protocols.bgp.operator.views.base import \
from os_ken.services.protocols.bgp.operator.views.base import OperatorDetailView
from os_ken.services.protocols.bgp.operator.views import fields
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_LOCAL_PREF
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_EXTENDED_COMMUNITIES
class NlriDetailView(OperatorDetailView):

    def __new__(cls, obj, filter_func=None):
        from os_ken.lib.packet.bgp import LabelledVPNIPAddrPrefix
        from os_ken.lib.packet.bgp import LabelledVPNIP6AddrPrefix
        from os_ken.lib.packet.bgp import IPAddrPrefix, IP6AddrPrefix
        if isinstance(obj, (LabelledVPNIPAddrPrefix, LabelledVPNIP6AddrPrefix)):
            return VpnNlriDetailView(obj)
        elif isinstance(obj, (IPAddrPrefix, IP6AddrPrefix)):
            return IpNlriDetailView(obj)
        else:
            return OperatorDetailView(obj, filter_func)

    def encode(self):
        return self._obj.formatted_nlri_str