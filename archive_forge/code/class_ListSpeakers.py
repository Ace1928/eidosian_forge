from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.common import validators
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.bgp import peer as bgp_peer
class ListSpeakers(neutronv20.ListCommand):
    """List BGP speakers."""
    resource = 'bgp_speaker'
    list_columns = ['id', 'name', 'local_as', 'ip_version']
    pagination_support = True
    sorting_support = True