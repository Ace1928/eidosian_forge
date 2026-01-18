import logging
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
class ListExtension(command.Lister):
    _description = _('List API extensions')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--compute', action='store_true', default=False, help=_('List extensions for the Compute API'))
        parser.add_argument('--identity', action='store_true', default=False, help=_('List extensions for the Identity API'))
        parser.add_argument('--network', action='store_true', default=False, help=_('List extensions for the Network API'))
        parser.add_argument('--volume', action='store_true', default=False, help=_('List extensions for the Block Storage API'))
        parser.add_argument('--long', action='store_true', default=False, help=_('List additional fields in output'))
        return parser

    def take_action(self, parsed_args):
        if parsed_args.long:
            columns = ('Name', 'Alias', 'Description', 'Namespace', 'Updated At', 'Links')
        else:
            columns = ('Name', 'Alias', 'Description')
        data = []
        show_all = not parsed_args.identity and (not parsed_args.compute) and (not parsed_args.volume) and (not parsed_args.network)
        if parsed_args.identity or show_all:
            identity_client = self.app.client_manager.identity
            try:
                data += identity_client.extensions.list()
            except Exception:
                message = _('Extensions list not supported by Identity API')
                LOG.warning(message)
        if parsed_args.compute or show_all:
            compute_client = self.app.client_manager.sdk_connection.compute
            try:
                data += compute_client.extensions()
            except Exception:
                message = _('Extensions list not supported by Compute API')
                LOG.warning(message)
        if parsed_args.volume or show_all:
            volume_client = self.app.client_manager.sdk_connection.volume
            try:
                data += volume_client.extensions()
            except Exception:
                message = _('Extensions list not supported by Block Storage API')
                LOG.warning(message)
        if parsed_args.network or show_all:
            network_client = self.app.client_manager.network
            try:
                data += network_client.extensions()
            except Exception:
                message = _('Failed to retrieve extensions list from Network API')
                LOG.warning(message)
        extension_tuples = (utils.get_item_properties(s, columns) for s in data)
        return (columns, extension_tuples)