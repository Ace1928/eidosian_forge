import logging
from osc_lib.command import command
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class CreateTLDCommand(command.ShowOne):
    """Create new tld"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--name', help='TLD Name', required=True)
        parser.add_argument('--description', help='Description')
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        data = client.tlds.create(parsed_args.name, parsed_args.description)
        _format_tld(data)
        return zip(*sorted(data.items()))