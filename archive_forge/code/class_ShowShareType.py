import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import constants
from manilaclient.osc import utils
class ShowShareType(command.ShowOne):
    """Show a share type."""
    _description = _('Display share type details')

    def get_parser(self, prog_name):
        parser = super(ShowShareType, self).get_parser(prog_name)
        parser.add_argument('share_type', metavar='<share_type>', help=_('Share type to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_type = apiutils.find_resource(share_client.share_types, parsed_args.share_type)
        formatted_type = format_share_type(share_type, parsed_args.formatter)
        return (ATTRIBUTES, oscutils.get_dict_properties(formatted_type._info, ATTRIBUTES))