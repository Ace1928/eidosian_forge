import logging
from cinderclient import api_versions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListTransferRequest(command.Lister):
    _description = _('Lists all volume transfer requests.')

    def get_parser(self, prog_name):
        parser = super(ListTransferRequest, self).get_parser(prog_name)
        parser.add_argument('--all-projects', dest='all_projects', action='store_true', default=False, help=_('Include all projects (admin only)'))
        return parser

    def take_action(self, parsed_args):
        columns = ['ID', 'Name', 'Volume ID']
        column_headers = ['ID', 'Name', 'Volume']
        volume_client = self.app.client_manager.volume
        volume_transfer_result = volume_client.transfers.list(detailed=True, search_opts={'all_tenants': parsed_args.all_projects})
        return (column_headers, (utils.get_item_properties(s, columns) for s in volume_transfer_result))