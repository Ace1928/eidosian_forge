from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
class ShowConsoleURL(command.ShowOne):
    _description = _("Show server's remote console URL")

    def get_parser(self, prog_name):
        parser = super(ShowConsoleURL, self).get_parser(prog_name)
        parser.add_argument('server', metavar='<server>', help=_('Server to show URL (name or ID)'))
        type_group = parser.add_mutually_exclusive_group()
        type_group.add_argument('--novnc', dest='url_type', action='store_const', const='novnc', default='novnc', help=_('Show noVNC console URL (default)'))
        type_group.add_argument('--xvpvnc', dest='url_type', action='store_const', const='xvpvnc', help=_('Show xvpvnc console URL'))
        type_group.add_argument('--spice', dest='url_type', action='store_const', const='spice-html5', help=_('Show SPICE console URL'))
        type_group.add_argument('--rdp', dest='url_type', action='store_const', const='rdp-html5', help=_('Show RDP console URL'))
        type_group.add_argument('--serial', dest='url_type', action='store_const', const='serial', help=_('Show serial console URL'))
        type_group.add_argument('--mks', dest='url_type', action='store_const', const='webmks', help=_('Show WebMKS console URL'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        server = compute_client.find_server(parsed_args.server, ignore_missing=False)
        data = compute_client.create_console(server.id, console_type=parsed_args.url_type)
        display_columns, columns = _get_console_columns(data)
        data = utils.get_dict_properties(data, columns)
        return (display_columns, data)