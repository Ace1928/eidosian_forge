import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import identity as identity_utils
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from openstackclient.i18n import _
class ShowNetworkTrunk(command.ShowOne):
    """Show information of a given network trunk"""

    def get_parser(self, prog_name):
        parser = super(ShowNetworkTrunk, self).get_parser(prog_name)
        parser.add_argument('trunk', metavar='<trunk>', help=_('Trunk to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        trunk_id = client.find_trunk(parsed_args.trunk).id
        obj = client.get_trunk(trunk_id)
        display_columns, columns = _get_columns(obj)
        data = osc_utils.get_dict_properties(obj, columns, formatters=_formatters)
        return (display_columns, data)