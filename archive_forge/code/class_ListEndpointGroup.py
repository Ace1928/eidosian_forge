from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronv20
class ListEndpointGroup(neutronv20.ListCommand):
    """List VPN endpoint groups that belong to a given tenant."""
    resource = 'endpoint_group'
    list_columns = ['id', 'name', 'type', 'endpoints']
    _formatters = {}
    pagination_support = True
    sorting_support = True