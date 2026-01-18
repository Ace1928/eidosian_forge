from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc.v2.networking_bgpvpn import constants
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
class DeleteBgpvpnNetAssoc(BgpvpnNetAssoc, DeleteBgpvpnResAssoc):
    _description = _('Delete a BGP VPN network association(s) for a given BGP VPN')
    pass