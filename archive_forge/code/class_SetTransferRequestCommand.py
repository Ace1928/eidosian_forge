import logging
from osc_lib.command import command
from osc_lib import exceptions as osc_exc
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class SetTransferRequestCommand(command.ShowOne):
    """Set a Zone Transfer Request"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('id', help='Zone Transfer Request ID')
        description_group = parser.add_mutually_exclusive_group()
        description_group.add_argument('--description', help='Description')
        description_group.add_argument('--no-description', action='store_true')
        parser.add_argument('--target-project-id', help='Target Project ID to transfer to.')
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        data = {}
        if parsed_args.no_description:
            data['description'] = None
        elif parsed_args.description:
            data['description'] = parsed_args.description
        if parsed_args.target_project_id:
            data['target_project_id'] = parsed_args.target_project_id
        updated = client.zone_transfers.update_request(parsed_args.id, data)
        return zip(*sorted(updated.items()))