from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.common import validators
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.bgp import peer as bgp_peer
class AddNetworkToSpeaker(neutronv20.NeutronCommand):
    """Add a network to the BGP speaker."""

    def get_parser(self, prog_name):
        parser = super(AddNetworkToSpeaker, self).get_parser(prog_name)
        parser.add_argument('bgp_speaker', metavar='BGP_SPEAKER', help=_('ID or name of the BGP speaker.'))
        parser.add_argument('network', metavar='NETWORK', help=_('ID or name of the network to add.'))
        return parser

    def take_action(self, parsed_args):
        neutron_client = self.get_client()
        _speaker_id = get_bgp_speaker_id(neutron_client, parsed_args.bgp_speaker)
        _net_id = get_network_id(neutron_client, parsed_args.network)
        neutron_client.add_network_to_bgp_speaker(_speaker_id, {'network_id': _net_id})
        print(_('Added network %(net)s to BGP speaker %(speaker)s.') % {'net': parsed_args.network, 'speaker': parsed_args.bgp_speaker}, file=self.app.stdout)