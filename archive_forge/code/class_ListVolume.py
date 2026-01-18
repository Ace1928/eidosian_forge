import argparse
import copy
import functools
import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class ListVolume(command.Lister):
    _description = _('List volumes')

    def get_parser(self, prog_name):
        parser = super(ListVolume, self).get_parser(prog_name)
        parser.add_argument('--project', metavar='<project>', help=_('Filter results by project (name or ID) (admin only)'))
        identity_common.add_project_domain_option_to_parser(parser)
        parser.add_argument('--user', metavar='<user>', help=_('Filter results by user (name or ID) (admin only)'))
        identity_common.add_user_domain_option_to_parser(parser)
        parser.add_argument('--name', metavar='<name>', help=_('Filter results by volume name'))
        parser.add_argument('--status', metavar='<status>', help=_('Filter results by status'))
        parser.add_argument('--all-projects', action='store_true', default=False, help=_('Include all projects (admin only)'))
        parser.add_argument('--long', action='store_true', default=False, help=_('List additional fields in output'))
        pagination.add_marker_pagination_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        identity_client = self.app.client_manager.identity
        if parsed_args.long:
            columns = ['ID', 'Name', 'Status', 'Size', 'Volume Type', 'Bootable', 'Attachments', 'Metadata']
            column_headers = copy.deepcopy(columns)
            column_headers[4] = 'Type'
            column_headers[6] = 'Attached to'
            column_headers[7] = 'Properties'
        else:
            columns = ['ID', 'Name', 'Status', 'Size', 'Attachments']
            column_headers = copy.deepcopy(columns)
            column_headers[4] = 'Attached to'
        server_cache = {}
        try:
            compute_client = self.app.client_manager.compute
            for s in compute_client.servers.list():
                server_cache[s.id] = s
        except Exception:
            pass
        AttachmentsColumnWithCache = functools.partial(AttachmentsColumn, server_cache=server_cache)
        project_id = None
        if parsed_args.project:
            project_id = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
        user_id = None
        if parsed_args.user:
            user_id = identity_common.find_user(identity_client, parsed_args.user, parsed_args.user_domain).id
        all_projects = bool(parsed_args.project) or parsed_args.all_projects
        search_opts = {'all_tenants': all_projects, 'project_id': project_id, 'user_id': user_id, 'name': parsed_args.name, 'status': parsed_args.status}
        data = volume_client.volumes.list(search_opts=search_opts, marker=parsed_args.marker, limit=parsed_args.limit)
        column_headers = utils.backward_compat_col_lister(column_headers, parsed_args.columns, {'Display Name': 'Name'})
        return (column_headers, (utils.get_item_properties(s, columns, formatters={'Metadata': format_columns.DictColumn, 'Attachments': AttachmentsColumnWithCache}) for s in data))