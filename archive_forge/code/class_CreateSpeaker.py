from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.common import validators
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.bgp import peer as bgp_peer
class CreateSpeaker(neutronv20.CreateCommand):
    """Create a BGP Speaker."""
    resource = 'bgp_speaker'

    def add_known_arguments(self, parser):
        parser.add_argument('name', metavar='NAME', help=_('Name of the BGP speaker to create.'))
        parser.add_argument('--local-as', metavar='LOCAL_AS', required=True, help=_('Local AS number. (Integer in [%(min_val)s, %(max_val)s] is allowed.)') % {'min_val': MIN_AS_NUM, 'max_val': MAX_AS_NUM})
        parser.add_argument('--ip-version', type=int, choices=[4, 6], default=4, help=_('IP version for the BGP speaker (default is 4).'))
        add_common_arguments(parser)

    def args2body(self, parsed_args):
        body = {}
        validate_speaker_attributes(parsed_args)
        body['local_as'] = parsed_args.local_as
        body['ip_version'] = parsed_args.ip_version
        args2body_common_arguments(body, parsed_args)
        return {self.resource: body}