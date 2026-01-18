from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import network
from neutronclient.neutron.v2_0 import router
class ListPoolsOnLbaasAgent(neutronV20.ListCommand):
    """List the pools on a loadbalancer agent."""
    list_columns = ['id', 'name', 'lb_method', 'protocol', 'admin_state_up', 'status']
    resource = 'pool'
    unknown_parts_flag = False

    def get_parser(self, prog_name):
        parser = super(ListPoolsOnLbaasAgent, self).get_parser(prog_name)
        parser.add_argument('lbaas_agent', metavar='LBAAS_AGENT', help=_('ID of the loadbalancer agent to query.'))
        return parser

    def call_server(self, neutron_client, search_opts, parsed_args):
        data = neutron_client.list_pools_on_lbaas_agent(parsed_args.lbaas_agent, **search_opts)
        return data