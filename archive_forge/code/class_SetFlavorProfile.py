from cliff import lister
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class SetFlavorProfile(command.Command):
    """Update a flavor profile"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('flavorprofile', metavar='<flavor_profile>', help='Name or UUID of the flavor profile to update.')
        parser.add_argument('--name', metavar='<name>', help='Set the name of the flavor profile.')
        parser.add_argument('--provider', metavar='<provider_name>', help='Set the provider of the flavor profile.')
        parser.add_argument('--flavor-data', metavar='<flavor_data>', help='Set the flavor data of the flavor profile.')
        return parser

    def take_action(self, parsed_args):
        attrs = v2_utils.get_flavorprofile_attrs(self.app.client_manager, parsed_args)
        flavorprofile_id = attrs.pop('flavorprofile_id')
        body = {'flavorprofile': attrs}
        self.app.client_manager.load_balancer.flavorprofile_set(flavorprofile_id, json=body)