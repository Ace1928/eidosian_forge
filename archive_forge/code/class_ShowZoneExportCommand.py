import logging
from osc_lib.command import command
from osc_lib import exceptions as osc_exc
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class ShowZoneExportCommand(command.ShowOne):
    """Show a Zone Export"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('zone_export_id', help='Zone Export ID', type=str)
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        data = client.zone_exports.get_export_record(parsed_args.zone_export_id)
        _format_zone_export_record(data)
        return zip(*sorted(data.items()))