import logging
from osc_lib.command import command
from osc_lib import exceptions as osc_exc
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class CreateTransferRequestCommand(command.ShowOne):
    """Create new zone transfer request"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('zone_id', help='Zone ID to transfer.')
        parser.add_argument('--target-project-id', help='Target Project ID to transfer to.')
        parser.add_argument('--description', help='Description')
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        data = client.zone_transfers.create_request(parsed_args.zone_id, parsed_args.target_project_id, parsed_args.description)
        return zip(*sorted(data.items()))