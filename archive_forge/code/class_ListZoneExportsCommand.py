import logging
from osc_lib.command import command
from osc_lib import exceptions as osc_exc
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class ListZoneExportsCommand(command.Lister):
    """List Zone Exports"""
    columns = ['id', 'zone_id', 'created_at', 'status']

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        data = client.zone_exports.list()
        cols = self.columns
        return (cols, (utils.get_item_properties(s, cols) for s in data['exports']))