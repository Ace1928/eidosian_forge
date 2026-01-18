import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
class ListFirewallRule(neutronv20.ListCommand):
    """List firewall rules that belong to a given tenant."""
    resource = 'firewall_rule'
    list_columns = ['id', 'name', 'firewall_policy_id', 'summary', 'enabled']
    pagination_support = True
    sorting_support = True

    def extend_list(self, data, parsed_args):
        for d in data:
            val = []
            if d.get('protocol'):
                protocol = d['protocol'].upper()
            else:
                protocol = 'no-protocol'
            val.append(protocol)
            if 'source_ip_address' in d and 'source_port' in d:
                src = 'source: ' + str(d['source_ip_address']).lower()
                src = src + '(' + str(d['source_port']).lower() + ')'
            else:
                src = 'source: none specified'
            val.append(src)
            if 'destination_ip_address' in d and 'destination_port' in d:
                dst = 'dest: ' + str(d['destination_ip_address']).lower()
                dst = dst + '(' + str(d['destination_port']).lower() + ')'
            else:
                dst = 'dest: none specified'
            val.append(dst)
            if 'action' in d:
                action = d['action']
            else:
                action = 'no-action'
            val.append(action)
            d['summary'] = ',\n '.join(val)