import uuid
from openstack import utils as sdk_utils
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class ListMigration(command.Lister):
    _description = _('List server migrations')

    def get_parser(self, prog_name):
        parser = super(ListMigration, self).get_parser(prog_name)
        parser.add_argument('--server', metavar='<server>', help=_('Filter migrations by server (name or ID)'))
        parser.add_argument('--host', metavar='<host>', help=_('Filter migrations by source or destination host'))
        parser.add_argument('--status', metavar='<status>', help=_('Filter migrations by status'))
        parser.add_argument('--type', metavar='<type>', choices=['evacuation', 'live-migration', 'cold-migration', 'resize'], help=_('Filter migrations by type'))
        pagination.add_marker_pagination_option_to_parser(parser)
        parser.add_argument('--changes-since', dest='changes_since', metavar='<changes-since>', help=_('List only migrations changed later or equal to a certain point of time. The provided time should be an ISO 8061 formatted time, e.g. ``2016-03-04T06:27:59Z``. (supported with --os-compute-api-version 2.59 or above)'))
        parser.add_argument('--changes-before', dest='changes_before', metavar='<changes-before>', help=_('List only migrations changed earlier or equal to a certain point of time. The provided time should be an ISO 8061 formatted time, e.g. ``2016-03-04T06:27:59Z``. (supported with --os-compute-api-version 2.66 or above)'))
        parser.add_argument('--project', metavar='<project>', help=_('Filter migrations by project (name or ID) (supported with --os-compute-api-version 2.80 or above)'))
        identity_common.add_project_domain_option_to_parser(parser)
        parser.add_argument('--user', metavar='<user>', help=_('Filter migrations by user (name or ID) (supported with --os-compute-api-version 2.80 or above)'))
        identity_common.add_user_domain_option_to_parser(parser)
        return parser

    def print_migrations(self, parsed_args, compute_client, migrations):
        column_headers = ['Source Node', 'Dest Node', 'Source Compute', 'Dest Compute', 'Dest Host', 'Status', 'Server UUID', 'Old Flavor', 'New Flavor', 'Created At', 'Updated At']
        columns = ['source_node', 'dest_node', 'source_compute', 'dest_compute', 'dest_host', 'status', 'server_id', 'old_flavor_id', 'new_flavor_id', 'created_at', 'updated_at']
        if sdk_utils.supports_microversion(compute_client, '2.59'):
            column_headers.insert(0, 'UUID')
            columns.insert(0, 'uuid')
        if sdk_utils.supports_microversion(compute_client, '2.23'):
            column_headers.insert(0, 'Id')
            columns.insert(0, 'id')
            column_headers.insert(len(column_headers) - 2, 'Type')
            columns.insert(len(columns) - 2, 'migration_type')
        if sdk_utils.supports_microversion(compute_client, '2.80'):
            if parsed_args.project:
                column_headers.insert(len(column_headers) - 2, 'Project')
                columns.insert(len(columns) - 2, 'project_id')
            if parsed_args.user:
                column_headers.insert(len(column_headers) - 2, 'User')
                columns.insert(len(columns) - 2, 'user_id')
        return (column_headers, (utils.get_item_properties(mig, columns) for mig in migrations))

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        identity_client = self.app.client_manager.identity
        search_opts = {}
        if parsed_args.host is not None:
            search_opts['host'] = parsed_args.host
        if parsed_args.status is not None:
            search_opts['status'] = parsed_args.status
        if parsed_args.server:
            server = compute_client.find_server(parsed_args.server, ignore_missing=False)
            search_opts['instance_uuid'] = server.id
        if parsed_args.type:
            migration_type = parsed_args.type
            if migration_type == 'cold-migration':
                migration_type = 'migration'
            search_opts['migration_type'] = migration_type
        if parsed_args.marker:
            if not sdk_utils.supports_microversion(compute_client, '2.59'):
                msg = _('--os-compute-api-version 2.59 or greater is required to support the --marker option')
                raise exceptions.CommandError(msg)
            search_opts['marker'] = parsed_args.marker
        if parsed_args.limit:
            if not sdk_utils.supports_microversion(compute_client, '2.59'):
                msg = _('--os-compute-api-version 2.59 or greater is required to support the --limit option')
                raise exceptions.CommandError(msg)
            search_opts['limit'] = parsed_args.limit
            search_opts['paginated'] = False
        if parsed_args.changes_since:
            if not sdk_utils.supports_microversion(compute_client, '2.59'):
                msg = _('--os-compute-api-version 2.59 or greater is required to support the --changes-since option')
                raise exceptions.CommandError(msg)
            search_opts['changes_since'] = parsed_args.changes_since
        if parsed_args.changes_before:
            if not sdk_utils.supports_microversion(compute_client, '2.66'):
                msg = _('--os-compute-api-version 2.66 or greater is required to support the --changes-before option')
                raise exceptions.CommandError(msg)
            search_opts['changes_before'] = parsed_args.changes_before
        if parsed_args.project:
            if not sdk_utils.supports_microversion(compute_client, '2.80'):
                msg = _('--os-compute-api-version 2.80 or greater is required to support the --project option')
                raise exceptions.CommandError(msg)
            search_opts['project_id'] = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
        if parsed_args.user:
            if not sdk_utils.supports_microversion(compute_client, '2.80'):
                msg = _('--os-compute-api-version 2.80 or greater is required to support the --user option')
                raise exceptions.CommandError(msg)
            search_opts['user_id'] = identity_common.find_user(identity_client, parsed_args.user, parsed_args.user_domain).id
        migrations = list(compute_client.migrations(**search_opts))
        return self.print_migrations(parsed_args, compute_client, migrations)