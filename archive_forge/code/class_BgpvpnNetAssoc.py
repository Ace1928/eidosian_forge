from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc.v2.networking_bgpvpn import constants
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
class BgpvpnNetAssoc(object):
    _assoc_res_name = constants.NETWORK_RESOURCE_NAME
    _resource = constants.NETWORK_ASSOC
    _resource_plural = constants.NETWORK_ASSOCS
    _attr_map = (('id', 'ID', column_util.LIST_BOTH), ('tenant_id', 'Project', column_util.LIST_LONG_ONLY), ('%s_id' % _assoc_res_name, '%s ID' % _assoc_res_name.capitalize(), column_util.LIST_BOTH))
    _formatters = {}