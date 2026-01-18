import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from oslo_utils import strutils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.osc import utils
class ShowShareGroupType(command.ShowOne):
    """Show Share Group Types."""
    _description = _('Show share group types')
    log = logging.getLogger(__name__ + '.ShowShareGroupType')

    def get_parser(self, prog_name):
        parser = super(ShowShareGroupType, self).get_parser(prog_name)
        parser.add_argument('share_group_type', metavar='<share-group-type>', help=_('Name or ID of the share group type to show'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_group_type = apiutils.find_resource(share_client.share_group_types, parsed_args.share_group_type)
        share_group_type_obj = share_client.share_group_types.get(share_group_type)
        formatter = parsed_args.formatter
        formatted_group_type = utils.format_share_group_type(share_group_type_obj, formatter)
        return (ATTRIBUTES, oscutils.get_dict_properties(formatted_group_type, ATTRIBUTES))