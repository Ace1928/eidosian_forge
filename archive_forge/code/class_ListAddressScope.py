from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ListAddressScope(neutronV20.ListCommand):
    """List address scopes that belong to a given tenant."""
    resource = 'address_scope'
    list_columns = ['id', 'name', 'ip_version']
    pagination_support = True
    sorting_support = True