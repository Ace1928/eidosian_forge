from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc.v2.networking_bgpvpn import constants
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
class ShowBgpvpnRouterAssoc(BgpvpnRouterAssoc, ShowBgpvpnResAssoc):
    _description = _('Show information of a given BGP VPN router association')
    pass