from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
class ListFirewall(neutronv20.ListCommand):
    """List firewalls that belong to a given tenant."""
    resource = 'firewall'
    list_columns = ['id', 'name', 'firewall_policy_id']
    _formatters = {}
    pagination_support = True
    sorting_support = True