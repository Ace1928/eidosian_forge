from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import network
from neutronclient.neutron.v2_0 import router
class GetLbaasAgentHostingLoadBalancer(neutronV20.ListCommand):
    """Get lbaas v2 agent hosting a loadbalancer.

    Deriving from ListCommand though server will return only one agent
    to keep common output format for all agent schedulers
    """
    resource = 'agent'
    list_columns = ['id', 'host', 'admin_state_up', 'alive']
    unknown_parts_flag = False

    def get_parser(self, prog_name):
        parser = super(GetLbaasAgentHostingLoadBalancer, self).get_parser(prog_name)
        parser.add_argument('loadbalancer', metavar='LOADBALANCER', help=_('LoadBalancer to query.'))
        return parser

    def extend_list(self, data, parsed_args):
        for agent in data:
            agent['alive'] = ':-)' if agent['alive'] else 'xxx'

    def call_server(self, neutron_client, search_opts, parsed_args):
        _id = neutronV20.find_resourceid_by_name_or_id(neutron_client, 'loadbalancer', parsed_args.loadbalancer)
        search_opts['loadbalancer'] = _id
        agent = neutron_client.get_lbaas_agent_hosting_loadbalancer(**search_opts)
        data = {'agents': [agent['agent']]}
        return data