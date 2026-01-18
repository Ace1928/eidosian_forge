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
class ListShareType(command.Lister):
    """List Share Types."""
    _description = _('List share types')

    def get_parser(self, prog_name):
        parser = super(ListShareType, self).get_parser(prog_name)
        parser.add_argument('--all', action='store_true', default=False, help=_('Display all share types whatever public or private. Default=False. (Admin only)'))
        parser.add_argument('--extra-specs', type=str, nargs='*', metavar='<key=value>', default=None, help=_('Filter share types with extra specs (key=value). Available only for API microversion >= 2.43. OPTIONAL: Default=None.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        search_opts = {}
        if parsed_args.extra_specs:
            if share_client.api_version < api_versions.APIVersion('2.43'):
                raise exceptions.CommandError("Filtering by 'extra_specs' is available only with API microversion '2.43' and above.")
            search_opts = {'extra_specs': utils.extract_extra_specs(extra_specs={}, specs_to_add=parsed_args.extra_specs)}
        share_types = share_client.share_types.list(search_opts=search_opts, show_all=parsed_args.all)
        formatted_types = []
        for share_type in share_types:
            formatted_types.append(format_share_type(share_type, parsed_args.formatter))
        values = (oscutils.get_dict_properties(s._info, ATTRIBUTES) for s in formatted_types)
        columns = utils.format_column_headers(ATTRIBUTES)
        return (columns, values)