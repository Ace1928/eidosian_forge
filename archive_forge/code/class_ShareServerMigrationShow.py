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
class ShareServerMigrationShow(command.ShowOne):
    """Obtains progress of share migration for a given share server.

    (Admin only, Experimental).

    :param share_server: either share_server object or text with its ID.

    """
    _description = _('Gets migration progress of a given share server when copying')

    def get_parser(self, prog_name):
        parser = super(ShareServerMigrationShow, self).get_parser(prog_name)
        parser.add_argument('share_server', metavar='<share_server>', help='ID of share server to show migration progress for.')
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        if share_client.api_version >= api_versions.APIVersion('2.57'):
            share_server = osc_utils.find_resource(share_client.share_servers, parsed_args.share_server)
            result = share_server.migration_get_progress()
            return self.dict2columns(result)
        else:
            raise exceptions.CommandError('Share Server Migration show is only available with manila API version >= 2.57')