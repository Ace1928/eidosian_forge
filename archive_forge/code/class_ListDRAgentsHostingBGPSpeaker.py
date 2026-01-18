from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0.bgp import speaker as bgp_speaker
class ListDRAgentsHostingBGPSpeaker(neutronV20.ListCommand):
    """List Dynamic Routing agents hosting a BGP speaker."""
    resource = 'agent'
    _formatters = {}
    list_columns = ['id', 'host', 'admin_state_up', 'alive']
    unknown_parts_flag = False

    def get_parser(self, prog_name):
        parser = super(ListDRAgentsHostingBGPSpeaker, self).get_parser(prog_name)
        parser.add_argument('bgp_speaker', metavar='BGP_SPEAKER', help=_('ID or name of the BGP speaker.'))
        return parser

    def extend_list(self, data, parsed_args):
        for agent in data:
            agent['alive'] = ':-)' if agent['alive'] else 'xxx'

    def call_server(self, neutron_client, search_opts, parsed_args):
        _speaker_id = bgp_speaker.get_bgp_speaker_id(neutron_client, parsed_args.bgp_speaker)
        search_opts['bgp_speaker'] = _speaker_id
        data = neutron_client.list_dragents_hosting_bgp_speaker(**search_opts)
        return data