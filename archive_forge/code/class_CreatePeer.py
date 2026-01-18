from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.common import validators
from neutronclient.neutron import v2_0 as neutronv20
class CreatePeer(neutronv20.CreateCommand):
    """Create a BGP Peer."""
    resource = 'bgp_peer'

    def add_known_arguments(self, parser):
        parser.add_argument('name', metavar='NAME', help=_('Name of the BGP peer to create.'))
        parser.add_argument('--peer-ip', metavar='PEER_IP_ADDRESS', required=True, help=_('Peer IP address.'))
        parser.add_argument('--remote-as', required=True, metavar='PEER_REMOTE_AS', help=_('Peer AS number. (Integer in [%(min_val)s, %(max_val)s] is allowed.)') % {'min_val': neutronv20.bgp.speaker.MIN_AS_NUM, 'max_val': neutronv20.bgp.speaker.MAX_AS_NUM})
        parser.add_argument('--auth-type', metavar='PEER_AUTH_TYPE', choices=['none', 'md5'], default='none', type=utils.convert_to_lowercase, help=_('Authentication algorithm. Supported algorithms: none(default), md5'))
        parser.add_argument('--password', metavar='AUTH_PASSWORD', help=_('Authentication password.'))

    def args2body(self, parsed_args):
        body = {}
        validate_peer_attributes(parsed_args)
        neutronv20.update_dict(parsed_args, body, ['name', 'peer_ip', 'remote_as', 'auth_type', 'password'])
        return {self.resource: body}