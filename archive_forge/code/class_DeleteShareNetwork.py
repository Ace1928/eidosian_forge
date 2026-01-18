import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common import cliutils
class DeleteShareNetwork(command.Command):
    """Delete one or more share networks"""
    _description = _('Delete one or more share networks')

    def get_parser(self, prog_name):
        parser = super(DeleteShareNetwork, self).get_parser(prog_name)
        parser.add_argument('share_network', metavar='<share-network>', nargs='+', help=_('Name or ID of the share network(s) to delete'))
        parser.add_argument('--wait', action='store_true', default=False, help=_('Wait for the share network(s) to be deleted'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        result = 0
        for share_network in parsed_args.share_network:
            try:
                share_network_obj = oscutils.find_resource(share_client.share_networks, share_network)
                share_client.share_networks.delete(share_network_obj)
                if parsed_args.wait:
                    if not oscutils.wait_for_delete(manager=share_client.share_networks, res_id=share_network_obj.id):
                        result += 1
            except Exception as e:
                result += 1
                LOG.error(f'Failed to delete share network with name or ID {share_network}: {e}')
        if result > 0:
            total = len(parsed_args.share_network)
            msg = f'{result} of {total} share networks failed to be deleted.'
            raise exceptions.CommandError(msg)