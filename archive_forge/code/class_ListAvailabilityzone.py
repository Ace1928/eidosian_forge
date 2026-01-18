from cliff import lister
from osc_lib.command import command
from osc_lib import utils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class ListAvailabilityzone(lister.Lister):
    """List availability zones"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--name', metavar='<name>', help='List availability zones according to their name.')
        parser.add_argument('--availabilityzoneprofile', metavar='<availabilityzone_profile>', help='List availability zones according to their AZ profile.')
        admin_state_group = parser.add_mutually_exclusive_group()
        admin_state_group.add_argument('--enable', action='store_true', default=None, help='List enabled availability zones.')
        admin_state_group.add_argument('--disable', action='store_true', default=None, help='List disabled availability zones.')
        return parser

    def take_action(self, parsed_args):
        columns = const.AVAILABILITYZONE_COLUMNS
        attrs = v2_utils.get_availabilityzone_attrs(self.app.client_manager, parsed_args)
        data = self.app.client_manager.load_balancer.availabilityzone_list(**attrs)
        formatters = {'availabilityzoneprofiles': v2_utils.format_list}
        return (columns, (utils.get_dict_properties(s, columns, formatters=formatters) for s in data['availability_zones']))