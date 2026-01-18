import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ListSubnet(neutronV20.ListCommand):
    """List subnets that belong to a given tenant."""
    resource = 'subnet'
    _formatters = {'allocation_pools': _format_allocation_pools, 'dns_nameservers': _format_dns_nameservers, 'host_routes': _format_host_routes}
    list_columns = ['id', 'name', 'cidr', 'allocation_pools']
    pagination_support = True
    sorting_support = True