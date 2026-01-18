import logging
from osc_lib.command import command
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class ListTSIGKeysCommand(command.Lister):
    """List tsigkeys"""
    columns = ['id', 'name', 'algorithm', 'secret', 'scope', 'resource_id']

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--name', help='TSIGKey NAME', required=False)
        parser.add_argument('--algorithm', help='TSIGKey algorithm', required=False)
        parser.add_argument('--scope', help='TSIGKey scope', required=False)
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        criterion = {}
        if parsed_args.name is not None:
            criterion['name'] = parsed_args.name
        if parsed_args.algorithm is not None:
            criterion['algorithm'] = parsed_args.algorithm
        if parsed_args.scope is not None:
            criterion['scope'] = parsed_args.scope
        data = get_all(client.tsigkeys.list, criterion)
        cols = self.columns
        return (cols, (utils.get_item_properties(s, cols) for s in data))