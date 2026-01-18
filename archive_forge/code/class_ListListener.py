from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ListListener(neutronV20.ListCommand):
    """LBaaS v2 List listeners that belong to a given tenant."""
    resource = 'listener'
    list_columns = ['id', 'default_pool_id', 'name', 'protocol', 'protocol_port', 'admin_state_up', 'status']
    pagination_support = True
    sorting_support = True