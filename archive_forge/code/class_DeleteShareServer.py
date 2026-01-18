import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import cliutils
from manilaclient.common import constants
class DeleteShareServer(command.Command):
    """Delete one or more share servers (Admin only)"""
    _description = _('Delete one or more share servers')

    def get_parser(self, prog_name):
        parser = super(DeleteShareServer, self).get_parser(prog_name)
        parser.add_argument('share_servers', metavar='<share-server>', nargs='+', help=_('ID(s) of the server(s) to delete'))
        parser.add_argument('--wait', action='store_true', default=False, help=_('Wait for share server deletion.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        result = 0
        for server in parsed_args.share_servers:
            try:
                server_obj = osc_utils.find_resource(share_client.share_servers, server)
                share_client.share_servers.delete(server_obj)
                if parsed_args.wait:
                    if not osc_utils.wait_for_delete(manager=share_client.share_servers, res_id=server_obj.id):
                        result += 1
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete a share server with ID '%(server)s': %(e)s"), {'server': server, 'e': e})
        if result > 0:
            total = len(parsed_args.share_servers)
            msg = f'Failed to delete {result} servers out of {total}.'
            raise exceptions.CommandError(_(msg))