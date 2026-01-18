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
class ShowShareProperties(command.ShowOne):
    """Show properties of a share"""
    _description = _('Show share properties')

    def get_parser(self, prog_name):
        parser = super(ShowShareProperties, self).get_parser(prog_name)
        parser.add_argument('share', metavar='<share>', help=_('Name or ID of share'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_obj = apiutils.find_resource(share_client.shares, parsed_args.share)
        share_properties = share_client.shares.get_metadata(share_obj)
        return self.dict2columns(share_properties._info)