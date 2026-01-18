import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
class ListFirewallPolicy(neutronv20.ListCommand):
    """List firewall policies that belong to a given tenant."""
    resource = 'firewall_policy'
    list_columns = ['id', 'name', 'firewall_rules']
    _formatters = {'firewall_rules': _format_firewall_rules}
    pagination_support = True
    sorting_support = True