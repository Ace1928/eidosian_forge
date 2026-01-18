from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ListRBACPolicy(neutronV20.ListCommand):
    """List RBAC policies that belong to a given tenant."""
    resource = 'rbac_policy'
    list_columns = ['id', 'object_type', 'object_id']
    pagination_support = True
    sorting_support = True
    allow_names = False