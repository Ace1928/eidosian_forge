import logging
from osc_lib.command import command
from osc_lib import exceptions as osc_exc
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class ImportZoneCommand(command.ShowOne):
    """Import a Zone from a file on the filesystem"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('zone_file_path', help='Path to a zone file', type=str)
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        with open(parsed_args.zone_file_path) as f:
            zone_file_contents = f.read()
        data = client.zone_imports.create(zone_file_contents)
        _format_zone_import_record(data)
        LOG.info('Zone Import %s was created', data['id'])
        return zip(*sorted(data.items()))