import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class CreateLocalIP(command.ShowOne):
    _description = _('Create Local IP')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--name', metavar='<name>', help=_('New local IP name'))
        parser.add_argument('--description', metavar='<description>', help=_('New local IP description'))
        parser.add_argument('--network', metavar='<network>', help=_('Network to allocate Local IP (name or ID)'))
        parser.add_argument('--local-port', metavar='<local-port>', help=_('Port to allocate Local IP (name or ID)'))
        parser.add_argument('--local-ip-address', metavar='<local-ip-address>', help=_('IP address or CIDR '))
        parser.add_argument('--ip-mode', metavar='<ip-mode>', help=_('local IP ip mode'))
        identity_common.add_project_domain_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = _get_attrs(self.app.client_manager, parsed_args)
        obj = client.create_local_ip(**attrs)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns, formatters={})
        return (display_columns, data)