from cliff import lister
from osc_lib.command import command
from osc_lib import utils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class ShowAvailabilityzoneProfile(command.ShowOne):
    """Show the details of a single availability zone profile"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('availabilityzoneprofile', metavar='<availabilityzone_profile>', help='Name or UUID of the availability zone profile to show.')
        return parser

    def take_action(self, parsed_args):
        rows = const.AVAILABILITYZONEPROFILE_ROWS
        attrs = v2_utils.get_availabilityzoneprofile_attrs(self.app.client_manager, parsed_args)
        availabilityzoneprofile_id = attrs.pop('availability_zone_profile_id')
        client_manager = self.app.client_manager
        data = client_manager.load_balancer.availabilityzoneprofile_show(availabilityzoneprofile_id=availabilityzoneprofile_id)
        return (rows, utils.get_dict_properties(data, rows, formatters={}))