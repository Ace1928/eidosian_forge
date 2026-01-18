from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.common import validators
from neutronclient.neutron import v2_0 as neutronv20
class UpdatePeer(neutronv20.UpdateCommand):
    """Update BGP Peer's information."""
    resource = 'bgp_peer'

    def add_known_arguments(self, parser):
        parser.add_argument('--name', help=_('Updated name of the BGP peer.'))
        parser.add_argument('--password', metavar='AUTH_PASSWORD', help=_('Updated authentication password.'))

    def args2body(self, parsed_args):
        body = {}
        neutronv20.update_dict(parsed_args, body, ['name', 'password'])
        return {self.resource: body}