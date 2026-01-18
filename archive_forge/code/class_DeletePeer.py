from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.common import validators
from neutronclient.neutron import v2_0 as neutronv20
class DeletePeer(neutronv20.DeleteCommand):
    """Delete a BGP peer."""
    resource = 'bgp_peer'