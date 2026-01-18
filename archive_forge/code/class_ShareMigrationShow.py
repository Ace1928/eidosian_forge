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
class ShareMigrationShow(command.ShowOne):
    """Gets migration progress of a given share when copying

    (Admin only, Experimental).

    """
    _description = _('Gets migration progress of a given share when copying')

    def get_parser(self, prog_name):
        parser = super(ShareMigrationShow, self).get_parser(prog_name)
        parser.add_argument('share', metavar='<share>', help=_('Name or ID of the share to get share migration progress information.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share = apiutils.find_resource(share_client.shares, parsed_args.share)
        result = share.migration_get_progress()
        return self.dict2columns(result[1])