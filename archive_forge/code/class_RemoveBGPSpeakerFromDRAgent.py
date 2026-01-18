from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0.bgp import speaker as bgp_speaker
class RemoveBGPSpeakerFromDRAgent(neutronV20.NeutronCommand):
    """Removes a BGP speaker from a Dynamic Routing agent."""

    def get_parser(self, prog_name):
        parser = super(RemoveBGPSpeakerFromDRAgent, self).get_parser(prog_name)
        add_common_args(parser)
        return parser

    def take_action(self, parsed_args):
        neutron_client = self.get_client()
        _speaker_id = bgp_speaker.get_bgp_speaker_id(neutron_client, parsed_args.bgp_speaker)
        neutron_client.remove_bgp_speaker_from_dragent(parsed_args.dragent_id, _speaker_id)
        print(_('Disassociated BGP speaker %s from the Dynamic Routing agent.') % parsed_args.bgp_speaker, file=self.app.stdout)