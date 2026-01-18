from os_ken.services.protocols.bgp.operator.views.base import \
from os_ken.services.protocols.bgp.operator.views.base import \
from os_ken.services.protocols.bgp.operator.views.base import OperatorDetailView
from os_ken.services.protocols.bgp.operator.views import fields
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_LOCAL_PREF
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_EXTENDED_COMMUNITIES
class TableDetailView(OperatorDetailView):
    scope_id = fields.DataField('scope_id')
    route_family = fields.DataField('route_family')
    destinations = fields.RelatedDictViewField('_destinations', 'os_ken.services.protocols.bgp.operator.views.bgp.DestinationDictView')