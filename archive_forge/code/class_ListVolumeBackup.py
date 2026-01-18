import copy
import functools
import logging
from cinderclient import api_versions
from cliff import columns as cliff_columns
from openstack import utils as sdk_utils
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
class ListVolumeBackup(command.Lister):
    _description = _('List volume backups')

    def get_parser(self, prog_name):
        parser = super(ListVolumeBackup, self).get_parser(prog_name)
        parser.add_argument('--long', action='store_true', default=False, help=_('List additional fields in output'))
        parser.add_argument('--name', metavar='<name>', help=_('Filters results by the backup name'))
        parser.add_argument('--status', metavar='<status>', choices=['creating', 'available', 'deleting', 'error', 'restoring', 'error_restoring'], help=_('Filters results by the backup status, one of: creating, available, deleting, error, restoring or error_restoring'))
        parser.add_argument('--volume', metavar='<volume>', help=_('Filters results by the volume which they backup (name or ID)'))
        pagination.add_marker_pagination_option_to_parser(parser)
        parser.add_argument('--all-projects', action='store_true', default=False, help=_('Include all projects (admin only)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.sdk_connection.volume
        columns = ('id', 'name', 'description', 'status', 'size', 'is_incremental')
        column_headers = ('ID', 'Name', 'Description', 'Status', 'Size', 'Incremental')
        if parsed_args.long:
            columns += ('availability_zone', 'volume_id', 'container')
            column_headers += ('Availability Zone', 'Volume', 'Container')
        volume_cache = {}
        try:
            for s in volume_client.volumes():
                volume_cache[s.id] = s
        except Exception:
            pass
        _VolumeIdColumn = functools.partial(VolumeIdColumn, volume_cache=volume_cache)
        filter_volume_id = None
        if parsed_args.volume:
            try:
                filter_volume_id = volume_client.find_volume(parsed_args.volume, ignore_missing=False).id
            except exceptions.CommandError:
                LOG.debug('No volume with ID %s existing, continuing to search for backups for that volume ID', parsed_args.volume)
                filter_volume_id = parsed_args.volume
        marker_backup_id = None
        if parsed_args.marker:
            marker_backup_id = volume_client.find_backup(parsed_args.marker, ignore_missing=False).id
        data = volume_client.backups(name=parsed_args.name, status=parsed_args.status, volume_id=filter_volume_id, all_tenants=parsed_args.all_projects, marker=marker_backup_id, limit=parsed_args.limit)
        return (column_headers, (utils.get_item_properties(s, columns, formatters={'volume_id': _VolumeIdColumn}) for s in data))