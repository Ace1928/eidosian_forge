from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ListLoadBalancer(neutronV20.ListCommand):
    """LBaaS v2 List loadbalancers that belong to a given tenant."""
    resource = 'loadbalancer'
    list_columns = ['id', 'name', 'vip_address', 'provisioning_status', 'provider']
    pagination_support = True
    sorting_support = True