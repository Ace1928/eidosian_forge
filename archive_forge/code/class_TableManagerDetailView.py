from os_ken.services.protocols.bgp.operator.views.base import \
from os_ken.services.protocols.bgp.operator.views.base import \
from os_ken.services.protocols.bgp.operator.views.base import OperatorDetailView
from os_ken.services.protocols.bgp.operator.views import fields
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_LOCAL_PREF
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_EXTENDED_COMMUNITIES
class TableManagerDetailView(OperatorDetailView):
    tables = fields.RelatedDictViewField('_tables', 'os_ken.services.protocols.bgp.operator.views.bgp.TableDictView')
    tables_for_rt = fields.RelatedDictViewField('_tables_for_rt', 'os_ken.services.protocols.bgp.operator.views.bgp.TableDictView')
    global_tables = fields.RelatedDictViewField('_global_tables', 'os_ken.services.protocols.bgp.operator.views.bgp.TableDictView')
    asbr_label_range = fields.DataField('_asbr_label_range')
    next_hop_label = fields.DataField('_next_hop_label')
    next_vpnv4_label = fields.DataField('_next_vpnv4_label')