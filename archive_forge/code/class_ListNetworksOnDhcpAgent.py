from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import network
from neutronclient.neutron.v2_0 import router
class ListNetworksOnDhcpAgent(network.ListNetwork):
    """List the networks on a DHCP agent."""
    unknown_parts_flag = False

    def get_parser(self, prog_name):
        parser = super(ListNetworksOnDhcpAgent, self).get_parser(prog_name)
        parser.add_argument('dhcp_agent', metavar='DHCP_AGENT', help=_('ID of the DHCP agent.'))
        return parser

    def call_server(self, neutron_client, search_opts, parsed_args):
        data = neutron_client.list_networks_on_dhcp_agent(parsed_args.dhcp_agent, **search_opts)
        return data