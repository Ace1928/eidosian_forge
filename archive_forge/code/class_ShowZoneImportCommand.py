import logging
from osc_lib.command import command
from osc_lib import exceptions as osc_exc
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class ShowZoneImportCommand(command.ShowOne):
    """Show a Zone Import"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('zone_import_id', help='Zone Import ID', type=str)
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        data = client.zone_imports.get_import_record(parsed_args.zone_import_id)
        _format_zone_import_record(data)
        return zip(*sorted(data.items()))