import argparse
from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import dns
class ListFloatingIP(neutronV20.ListCommand):
    """List floating IPs that belong to a given tenant."""
    resource = 'floatingip'
    list_columns = ['id', 'fixed_ip_address', 'floating_ip_address', 'port_id']
    pagination_support = True
    sorting_support = True