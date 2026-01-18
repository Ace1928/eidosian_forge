from os_ken.services.protocols.bgp.operator.views.base import \
from os_ken.services.protocols.bgp.operator.views.base import \
from os_ken.services.protocols.bgp.operator.views.base import OperatorDetailView
from os_ken.services.protocols.bgp.operator.views import fields
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_LOCAL_PREF
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_EXTENDED_COMMUNITIES
class ReceivedRouteDetailView(OperatorDetailView):
    timestamp = fields.DataField('timestamp')
    filtered = fields.DataField('filtered')
    path = fields.RelatedViewField('path', 'os_ken.services.protocols.bgp.operator.views.bgp.PathDetailView')
    peer = fields.RelatedViewField('received_peer', 'os_ken.services.protocols.bgp.operator.views.bgp.PeerDetailView')

    def encode(self):
        ret = super(ReceivedRouteDetailView, self).encode()
        ret.update({'path': self.rel('path').encode()})
        return ret