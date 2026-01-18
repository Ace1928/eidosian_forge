import logging
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
class ListServerGroup(command.Lister):
    _description = _('List all server groups.')

    def get_parser(self, prog_name):
        parser = super(ListServerGroup, self).get_parser(prog_name)
        parser.add_argument('--all-projects', action='store_true', default=False, help=_('Display information from all projects (admin only)'))
        parser.add_argument('--long', action='store_true', default=False, help=_('List additional fields in output'))
        pagination.add_offset_pagination_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        kwargs = {}
        if parsed_args.all_projects:
            kwargs['all_projects'] = parsed_args.all_projects
        if parsed_args.offset:
            kwargs['offset'] = parsed_args.offset
        if parsed_args.limit:
            kwargs['limit'] = parsed_args.limit
        data = compute_client.server_groups(**kwargs)
        policy_key = 'Policies'
        if sdk_utils.supports_microversion(compute_client, '2.64'):
            policy_key = 'Policy'
        columns = ('id', 'name', policy_key.lower())
        column_headers = ('ID', 'Name', policy_key)
        if parsed_args.long:
            columns += ('member_ids', 'project_id', 'user_id')
            column_headers += ('Members', 'Project Id', 'User Id')
        return (column_headers, (utils.get_item_properties(s, columns, formatters=_formatters) for s in data))