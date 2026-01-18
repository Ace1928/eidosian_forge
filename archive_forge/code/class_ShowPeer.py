from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.common import validators
from neutronclient.neutron import v2_0 as neutronv20
class ShowPeer(neutronv20.ShowCommand):
    """Show information of a given BGP peer."""
    resource = 'bgp_peer'