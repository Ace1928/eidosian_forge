from cliff import lister
from osc_lib.command import command
from osc_lib import utils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class CreateAvailabilityzone(command.ShowOne):
    """Create an octavia availability zone"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--name', metavar='<name>', required=True, help='New availability zone name.')
        parser.add_argument('--availabilityzoneprofile', metavar='<availabilityzone_profile>', required=True, help='Availability zone profile to add the AZ to (name or ID).')
        parser.add_argument('--description', metavar='<description>', help='Set the availability zone description.')
        admin_group = parser.add_mutually_exclusive_group()
        admin_group.add_argument('--enable', action='store_true', default=None, help='Enable the availability zone.')
        admin_group.add_argument('--disable', action='store_true', default=None, help='Disable the availability zone.')
        return parser

    def take_action(self, parsed_args):
        rows = const.AVAILABILITYZONE_ROWS
        attrs = v2_utils.get_availabilityzone_attrs(self.app.client_manager, parsed_args)
        body = {'availability_zone': attrs}
        data = self.app.client_manager.load_balancer.availabilityzone_create(json=body)
        formatters = {'availability_zone_profiles': v2_utils.format_list}
        return (rows, utils.get_dict_properties(data['availability_zone'], rows, formatters=formatters))