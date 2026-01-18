from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc.v2.networking_bgpvpn import constants
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
def _args2body(self, _, args):
    attrs = {'advertise_extra_routes': False}
    if args.advertise_extra_routes:
        attrs['advertise_extra_routes'] = self._action != 'unset'
    elif args.no_advertise_extra_routes:
        attrs['advertise_extra_routes'] = self._action == 'unset'
    return attrs