from cliff import lister
from osc_lib.command import command
from osc_lib import utils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class SetAvailabilityzoneProfile(command.Command):
    """Update an availability zone profile"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('availabilityzoneprofile', metavar='<availabilityzone_profile>', help='Name or UUID of the availability zone profile to update.')
        parser.add_argument('--name', metavar='<name>', help='Set the name of the availability zone profile.')
        parser.add_argument('--provider', metavar='<provider_name>', help='Set the provider of the availability zone profile.')
        parser.add_argument('--availability-zone-data', metavar='<availability_zone_data>', help='Set the availability zone data of the profile.')
        return parser

    def take_action(self, parsed_args):
        attrs = v2_utils.get_availabilityzoneprofile_attrs(self.app.client_manager, parsed_args)
        availabilityzoneprofile_id = attrs.pop('availability_zone_profile_id')
        body = {'availability_zone_profile': attrs}
        self.app.client_manager.load_balancer.availabilityzoneprofile_set(availabilityzoneprofile_id, json=body)