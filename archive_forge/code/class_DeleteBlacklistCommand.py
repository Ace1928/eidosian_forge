import logging
from osc_lib.command import command
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class DeleteBlacklistCommand(command.Command):
    """Delete blacklist"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('id', help='Blacklist ID')
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        client.blacklists.delete(parsed_args.id)
        LOG.info('Blacklist %s was deleted', parsed_args.id)