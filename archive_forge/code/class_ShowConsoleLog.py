from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
class ShowConsoleLog(command.Command):
    _description = _("Show server's console output")

    def get_parser(self, prog_name):
        parser = super(ShowConsoleLog, self).get_parser(prog_name)
        parser.add_argument('server', metavar='<server>', help=_('Server to show console log (name or ID)'))
        parser.add_argument('--lines', metavar='<num-lines>', type=int, default=None, action=parseractions.NonNegativeAction, help=_('Number of lines to display from the end of the log (default=all)'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        server = compute_client.find_server(name_or_id=parsed_args.server, ignore_missing=False)
        output = compute_client.get_server_console_output(server.id, length=parsed_args.lines)
        data = None
        if output:
            data = output.get('output', None)
        if data and data[-1] != '\n':
            data += '\n'
        self.app.stdout.write(data)