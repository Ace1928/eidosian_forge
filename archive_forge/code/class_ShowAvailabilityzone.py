from cliff import lister
from osc_lib.command import command
from osc_lib import utils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class ShowAvailabilityzone(command.ShowOne):
    """Show the details for a single availability zone"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('availabilityzone', metavar='<availabilityzone>', help='Name of the availability zone.')
        return parser

    def take_action(self, parsed_args):
        rows = const.AVAILABILITYZONE_ROWS
        attrs = v2_utils.get_availabilityzone_attrs(self.app.client_manager, parsed_args)
        availabilityzone_name = attrs.pop('availabilityzone_name')
        data = self.app.client_manager.load_balancer.availabilityzone_show(availabilityzone_name=availabilityzone_name)
        formatters = {'availabilityzoneprofiles': v2_utils.format_list}
        return (rows, utils.get_dict_properties(data, rows, formatters=formatters))