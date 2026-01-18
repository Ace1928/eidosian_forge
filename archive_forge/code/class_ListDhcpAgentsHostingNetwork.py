from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import network
from neutronclient.neutron.v2_0 import router
class ListDhcpAgentsHostingNetwork(neutronV20.ListCommand):
    """List DHCP agents hosting a network."""
    resource = 'agent'
    _formatters = {}
    list_columns = ['id', 'host', 'admin_state_up', 'alive']
    unknown_parts_flag = False

    def get_parser(self, prog_name):
        parser = super(ListDhcpAgentsHostingNetwork, self).get_parser(prog_name)
        parser.add_argument('network', metavar='NETWORK', help=_('Network to query.'))
        return parser

    def extend_list(self, data, parsed_args):
        for agent in data:
            agent['alive'] = ':-)' if agent['alive'] else 'xxx'

    def call_server(self, neutron_client, search_opts, parsed_args):
        _id = neutronV20.find_resourceid_by_name_or_id(neutron_client, 'network', parsed_args.network)
        search_opts['network'] = _id
        data = neutron_client.list_dhcp_agent_hosting_networks(**search_opts)
        return data