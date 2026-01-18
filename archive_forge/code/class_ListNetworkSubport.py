import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import identity as identity_utils
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from openstackclient.i18n import _
class ListNetworkSubport(command.Lister):
    """List all subports for a given network trunk"""

    def get_parser(self, prog_name):
        parser = super(ListNetworkSubport, self).get_parser(prog_name)
        parser.add_argument('--trunk', required=True, metavar='<trunk>', help=_('List subports belonging to this trunk (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        trunk_id = client.find_trunk(parsed_args.trunk)
        data = client.get_trunk_subports(trunk_id)
        headers = ('Port', 'Segmentation Type', 'Segmentation ID')
        columns = ('port_id', 'segmentation_type', 'segmentation_id')
        return (headers, (osc_utils.get_dict_properties(s, columns) for s in data[SUB_PORTS]))