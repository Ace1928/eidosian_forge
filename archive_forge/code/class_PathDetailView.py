from os_ken.services.protocols.bgp.operator.views.base import \
from os_ken.services.protocols.bgp.operator.views.base import \
from os_ken.services.protocols.bgp.operator.views.base import OperatorDetailView
from os_ken.services.protocols.bgp.operator.views import fields
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_LOCAL_PREF
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_EXTENDED_COMMUNITIES
class PathDetailView(OperatorDetailView):
    source_version_num = fields.DataField('source_version_num')
    route_family = fields.RelatedViewField('route_family', 'os_ken.services.protocols.bgp.operator.views.bgp.RouteFamilyView')
    nlri = fields.RelatedViewField('nlri', 'os_ken.services.protocols.bgp.operator.views.bgp.NlriDetailView')
    is_withdraw = fields.DataField('is_withdraw')
    nexthop = fields.DataField('nexthop')
    pathattr_map = fields.DataField('pathattr_map')
    source = fields.RelatedViewField('source', 'os_ken.services.protocols.bgp.operator.views.bgp.PeerDetailView')

    def encode(self):
        ret = super(PathDetailView, self).encode()
        ret['nlri'] = self.rel('nlri').encode()
        ret['route_family'] = self.rel('route_family').encode()
        as_path = self.get_field('pathattr_map').get(BGP_ATTR_TYPE_AS_PATH)
        origin = self.get_field('pathattr_map').get(BGP_ATTR_TYPE_ORIGIN)
        metric = self.get_field('pathattr_map').get(BGP_ATTR_TYPE_MULTI_EXIT_DISC)
        local_pref = self.get_field('pathattr_map').get(BGP_ATTR_TYPE_LOCAL_PREF)
        ret['as_path'] = as_path.value if as_path else None
        ret['origin'] = origin.value if origin else None
        ret['metric'] = metric.value if metric else None
        ret['local_pref'] = local_pref.value if local_pref else None
        ext = ret['pathattr_map'].get(BGP_ATTR_TYPE_EXTENDED_COMMUNITIES)
        del ret['pathattr_map']
        if ext:
            ret['rt_list'] = ext.rt_list
            ret['soo_list'] = ext.soo_list
        return ret