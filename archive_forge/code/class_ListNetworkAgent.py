import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListNetworkAgent(command.Lister):
    _description = _('List network agents')

    def get_parser(self, prog_name):
        parser = super(ListNetworkAgent, self).get_parser(prog_name)
        parser.add_argument('--agent-type', metavar='<agent-type>', choices=['bgp', 'dhcp', 'open-vswitch', 'linux-bridge', 'ofa', 'l3', 'loadbalancer', 'metering', 'metadata', 'macvtap', 'nic', 'baremetal'], help=_('List only agents with the specified agent type. The supported agent types are: bgp, dhcp, open-vswitch, linux-bridge, ofa, l3, loadbalancer, metering, metadata, macvtap, nic, baremetal.'))
        parser.add_argument('--host', metavar='<host>', help=_('List only agents running on the specified host'))
        agent_type_group = parser.add_mutually_exclusive_group()
        agent_type_group.add_argument('--network', metavar='<network>', help=_('List agents hosting a network (name or ID)'))
        agent_type_group.add_argument('--router', metavar='<router>', help=_('List agents hosting this router (name or ID)'))
        parser.add_argument('--long', action='store_true', default=False, help=_('List additional fields in output'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        columns = ('id', 'agent_type', 'host', 'availability_zone', 'is_alive', 'is_admin_state_up', 'binary')
        column_headers = ('ID', 'Agent Type', 'Host', 'Availability Zone', 'Alive', 'State', 'Binary')
        key_value = {'bgp': 'BGP dynamic routing agent', 'dhcp': 'DHCP agent', 'open-vswitch': 'Open vSwitch agent', 'linux-bridge': 'Linux bridge agent', 'ofa': 'OFA driver agent', 'l3': 'L3 agent', 'loadbalancer': 'Loadbalancer agent', 'metering': 'Metering agent', 'metadata': 'Metadata agent', 'macvtap': 'Macvtap agent', 'nic': 'NIC Switch agent', 'baremetal': 'Baremetal Node'}
        filters = {}
        if parsed_args.network is not None:
            network = client.find_network(parsed_args.network, ignore_missing=False)
            data = client.network_hosting_dhcp_agents(network)
        elif parsed_args.router is not None:
            if parsed_args.long:
                columns += ('ha_state',)
                column_headers += ('HA State',)
            router = client.find_router(parsed_args.router, ignore_missing=False)
            data = client.routers_hosting_l3_agents(router)
        else:
            if parsed_args.agent_type is not None:
                filters['agent_type'] = key_value[parsed_args.agent_type]
            if parsed_args.host is not None:
                filters['host'] = parsed_args.host
            data = client.agents(**filters)
        return (column_headers, (utils.get_item_properties(s, columns, formatters=_formatters) for s in data))