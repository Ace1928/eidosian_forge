import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class ShowLocalIP(command.ShowOne):
    _description = _('Display local IP details')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('local_ip', metavar='<local-ip>', help=_('Local IP to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_local_ip(parsed_args.local_ip, ignore_missing=False)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns, formatters={})
        return (display_columns, data)