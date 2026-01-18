import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.vpn import utils as vpn_utils
class ListIPsecPolicy(neutronv20.ListCommand):
    """List IPsec policies that belong to a given tenant connection."""
    resource = 'ipsecpolicy'
    list_columns = ['id', 'name', 'auth_algorithm', 'encryption_algorithm', 'pfs']
    _formatters = {}
    pagination_support = True
    sorting_support = True