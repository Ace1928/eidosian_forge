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
class ShareServerMigrationComplete(command.Command):
    """Completes migration for a given share server (Admin only, Experimental).

    """
    _description = _('Completes migration for a given share server')

    def get_parser(self, prog_name):
        parser = super(ShareServerMigrationComplete, self).get_parser(prog_name)
        parser.add_argument('share_server', metavar='<share_server>', help=_('ID of share server to complete migration.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_server = osc_utils.find_resource(share_client.share_servers, parsed_args.share_server)
        if share_client.api_version >= api_versions.APIVersion('2.57'):
            share_server.migration_complete()
        else:
            raise exceptions.CommandError('Share Server Migration complete is only available with manila API version >= 2.57')