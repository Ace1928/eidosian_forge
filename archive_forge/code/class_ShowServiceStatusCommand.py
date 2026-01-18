import logging
from osc_lib.command import command
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2 import utils as v2_utils
class ShowServiceStatusCommand(command.ShowOne):
    """Show service status details"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('id', help='Service Status ID')
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        data = client.service_statuses.get(parsed_args.id)
        _format_status(data)
        return zip(*sorted(data.items()))