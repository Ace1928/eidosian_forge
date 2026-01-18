import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import exceptions as apiclient_exceptions
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import cliutils
from manilaclient.osc import utils
class ShareMigrationCancel(command.Command):
    """Cancels migration of a given share when copying

    (Admin only, Experimental).

    """
    _description = _('Cancels migration of a given share when copying')

    def get_parser(self, prog_name):
        parser = super(ShareMigrationCancel, self).get_parser(prog_name)
        parser.add_argument('share', metavar='<share>', help=_('Name or ID of share to migrate.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share = apiutils.find_resource(share_client.shares, parsed_args.share)
        share.migration_cancel()