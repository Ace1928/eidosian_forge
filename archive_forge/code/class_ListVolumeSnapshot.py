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
class ListVolumeSnapshot(command.Lister):
    _description = _('List volume snapshots')

    def get_parser(self, prog_name):
        parser = super(ListVolumeSnapshot, self).get_parser(prog_name)
        parser.add_argument('--all-projects', action='store_true', default=False, help=_('Include all projects (admin only)'))
        parser.add_argument('--project', metavar='<project>', help=_('Filter results by project (name or ID) (admin only)'))
        identity_common.add_project_domain_option_to_parser(parser)
        parser.add_argument('--long', action='store_true', default=False, help=_('List additional fields in output'))
        parser.add_argument('--name', metavar='<name>', default=None, help=_('Filters results by a name.'))
        parser.add_argument('--status', metavar='<status>', choices=['available', 'error', 'creating', 'deleting', 'error_deleting'], help=_("Filters results by a status. ('available', 'error', 'creating', 'deleting' or 'error_deleting')"))
        parser.add_argument('--volume', metavar='<volume>', default=None, help=_('Filters results by a volume (name or ID).'))
        pagination.add_marker_pagination_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        identity_client = self.app.client_manager.identity
        if parsed_args.long:
            columns = ['ID', 'Name', 'Description', 'Status', 'Size', 'Created At', 'Volume ID', 'Metadata']
            column_headers = copy.deepcopy(columns)
            column_headers[6] = 'Volume'
            column_headers[7] = 'Properties'
        else:
            columns = ['ID', 'Name', 'Description', 'Status', 'Size']
            column_headers = copy.deepcopy(columns)
        volume_cache = {}
        try:
            for s in volume_client.volumes.list():
                volume_cache[s.id] = s
        except Exception:
            pass
        _VolumeIdColumn = functools.partial(VolumeIdColumn, volume_cache=volume_cache)
        volume_id = None
        if parsed_args.volume:
            volume_id = utils.find_resource(volume_client.volumes, parsed_args.volume).id
        project_id = None
        if parsed_args.project:
            project_id = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
        all_projects = True if parsed_args.project else parsed_args.all_projects
        search_opts = {'all_tenants': all_projects, 'project_id': project_id, 'name': parsed_args.name, 'status': parsed_args.status, 'volume_id': volume_id}
        data = volume_client.volume_snapshots.list(search_opts=search_opts, marker=parsed_args.marker, limit=parsed_args.limit)
        return (column_headers, (utils.get_item_properties(s, columns, formatters={'Metadata': format_columns.DictColumn, 'Volume ID': _VolumeIdColumn}) for s in data))