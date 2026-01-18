import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from oslo_utils import strutils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.osc import utils
class ListShareGroupType(command.Lister):
    """List Share Group Types."""
    _description = _('List share types')
    log = logging.getLogger(__name__ + '.ListShareGroupType')

    def get_parser(self, prog_name):
        parser = super(ListShareGroupType, self).get_parser(prog_name)
        parser.add_argument('--all', action='store_true', default=False, help=_('Display all share group types whether public or private. Default=False. (Admin only)'))
        parser.add_argument('--group-specs', type=str, nargs='*', metavar='<key=value>', default=None, help=_('Filter share group types with group specs (key=value).'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        search_opts = {}
        if parsed_args.group_specs:
            search_opts = {'group_specs': utils.extract_group_specs(extra_specs={}, specs_to_add=parsed_args.group_specs)}
        formatter = parsed_args.formatter
        share_group_types = share_client.share_group_types.list(search_opts=search_opts, show_all=parsed_args.all)
        formatted_types = []
        for share_group_type in share_group_types:
            formatted_types.append(utils.format_share_group_type(share_group_type, formatter))
        column_headers = utils.format_column_headers(ATTRIBUTES)
        values = (oscutils.get_dict_properties(sgt, ATTRIBUTES) for sgt in formatted_types)
        return (column_headers, values)