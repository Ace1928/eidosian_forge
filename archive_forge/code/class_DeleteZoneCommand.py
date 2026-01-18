import logging
from osc_lib.command import command
from osc_lib import exceptions as osc_exc
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class DeleteZoneCommand(command.ShowOne):
    """Delete zone"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('id', help='Zone ID')
        parser.add_argument('--delete-shares', default=False, action='store_true', help='Delete existing zone shares. Default: False')
        common.add_all_common_options(parser)
        common.add_hard_delete_option(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        delete_shares = False
        if hasattr(parsed_args, 'delete_shares') and parsed_args.delete_shares is not None and isinstance(parsed_args.delete_shares, bool):
            delete_shares = parsed_args.delete_shares
        data = client.zones.delete(parsed_args.id, delete_shares=delete_shares)
        LOG.info('Zone %s was deleted', parsed_args.id)
        _format_zone(data)
        return zip(*sorted(data.items()))