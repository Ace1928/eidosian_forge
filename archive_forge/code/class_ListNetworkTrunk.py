import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import identity as identity_utils
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from openstackclient.i18n import _
class ListNetworkTrunk(command.Lister):
    """List all network trunks"""

    def get_parser(self, prog_name):
        parser = super(ListNetworkTrunk, self).get_parser(prog_name)
        parser.add_argument('--long', action='store_true', default=False, help=_('List additional fields in output'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        data = client.trunks()
        headers = ('ID', 'Name', 'Parent Port', 'Description')
        columns = ('id', 'name', 'port_id', 'description')
        if parsed_args.long:
            headers += ('Status', 'State', 'Created At', 'Updated At')
            columns += ('status', 'admin_state_up', 'created_at', 'updated_at')
        return (headers, (osc_utils.get_item_properties(s, columns, formatters=_formatters) for s in data))