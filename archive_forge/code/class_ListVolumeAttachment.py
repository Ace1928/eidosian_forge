import logging
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class ListVolumeAttachment(command.Lister):
    """Lists all volume attachments."""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--project', dest='project', metavar='<project>', help=_('Filter results by project (name or ID) (admin only)'))
        identity_common.add_project_domain_option_to_parser(parser)
        parser.add_argument('--all-projects', dest='all_projects', action='store_true', default=utils.env('ALL_PROJECTS', default=False), help=_('Shows details for all projects (admin only).'))
        parser.add_argument('--volume-id', metavar='<volume-id>', default=None, help=_('Filters results by a volume ID. ') + _FILTER_DEPRECATED)
        parser.add_argument('--status', metavar='<status>', help=_('Filters results by a status. ') + _FILTER_DEPRECATED)
        pagination.add_marker_pagination_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        identity_client = self.app.client_manager.identity
        if volume_client.api_version < api_versions.APIVersion('3.27'):
            msg = _("--os-volume-api-version 3.27 or greater is required to support the 'volume attachment list' command")
            raise exceptions.CommandError(msg)
        project_id = None
        if parsed_args.project:
            project_id = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
        search_opts = {'all_tenants': True if project_id else parsed_args.all_projects, 'project_id': project_id, 'status': parsed_args.status, 'volume_id': parsed_args.volume_id}
        attachments = volume_client.attachments.list(search_opts=search_opts, marker=parsed_args.marker, limit=parsed_args.limit)
        column_headers = ('ID', 'Volume ID', 'Server ID', 'Status')
        columns = ('id', 'volume_id', 'instance', 'status')
        return (column_headers, (utils.get_item_properties(a, columns) for a in attachments))