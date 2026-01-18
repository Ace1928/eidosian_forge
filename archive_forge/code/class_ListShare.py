import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import exceptions as apiclient_exceptions
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import cliutils
from manilaclient.osc import utils
class ListShare(command.Lister):
    """List Shared file systems (shares)."""
    _description = _('List shares')

    def get_parser(self, prog_name):
        parser = super(ListShare, self).get_parser(prog_name)
        parser.add_argument('--name', metavar='<share-name>', help=_('Filter shares by share name'))
        parser.add_argument('--status', metavar='<share-status>', help=_('Filter shares by status'))
        parser.add_argument('--snapshot', metavar='<share-network-id>', help=_('Filter shares by snapshot name or id.'))
        parser.add_argument('--export-location', metavar='<export-location>', help=_('Filter shares by export location id or path. Available only for microversion >= 2.35'))
        parser.add_argument('--soft-deleted', action='store_true', help=_('Get shares in recycle bin. If this parameter is set to True (Default=False), only shares in the recycle bin will be displayed. Available only for microversion >= 2.69.'))
        parser.add_argument('--public', action='store_true', default=False, help=_('Include public shares'))
        parser.add_argument('--share-network', metavar='<share-network-name-or-id>', help=_('Filter shares exported on a given share network'))
        parser.add_argument('--share-type', metavar='<share-type-name-or-id>', help=_('Filter shares of a given share type'))
        parser.add_argument('--share-group', metavar='<share-group-name-or-id>', help=_('Filter shares belonging to a given share group'))
        parser.add_argument('--host', metavar='<share-host>', help=_('Filter shares belonging to a given host (admin only)'))
        parser.add_argument('--share-server', metavar='<share-server-id>', help=_('Filter shares exported via a given share server (admin only)'))
        parser.add_argument('--project', metavar='<project>', help=_('Filter shares by project (name or ID) (admin only)'))
        identity_common.add_project_domain_option_to_parser(parser)
        parser.add_argument('--user', metavar='<user>', help=_('Filter results by user (name or ID) (admin only)'))
        identity_common.add_user_domain_option_to_parser(parser)
        parser.add_argument('--all-projects', action='store_true', default=False, help=_('Include all projects (admin only)'))
        parser.add_argument('--property', metavar='<key=value>', action=parseractions.KeyValueAction, help=_('Filter shares having a given metadata key=value property (repeat option to filter by multiple properties)'))
        parser.add_argument('--extra-spec', metavar='<key=value>', action=parseractions.KeyValueAction, help=_('Filter shares with extra specs (key=value) of the share type that they belong to. (repeat option to filter by multiple extra specs)'))
        parser.add_argument('--long', action='store_true', default=False, help=_('List additional fields in output'))
        parser.add_argument('--sort', metavar='<key>[:<direction>]', default='name:asc', help=_('Sort output by selected keys and directions(asc or desc) (default: name:asc), multiple keys and directions can be specified separated by comma'))
        parser.add_argument('--limit', metavar='<num-shares>', type=int, action=parseractions.NonNegativeAction, help=_('Maximum number of shares to display'))
        parser.add_argument('--marker', metavar='<share>', help=_('The last share ID of the previous page'))
        parser.add_argument('--name~', metavar='<name~>', default=None, help=_('Filter results matching a share name pattern. Available only for microversion >= 2.36.'))
        parser.add_argument('--description~', metavar='<description~>', default=None, help=_('Filter results matching a share description pattern.Available only for microversion >= 2.36.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        identity_client = self.app.client_manager.identity
        if parsed_args.long:
            columns = SHARE_ATTRIBUTES
            column_headers = SHARE_ATTRIBUTES_HEADERS
        else:
            columns = ['id', 'name', 'size', 'share_proto', 'status', 'is_public', 'share_type_name', 'host', 'availability_zone']
            column_headers = ['ID', 'Name', 'Size', 'Share Proto', 'Status', 'Is Public', 'Share Type Name', 'Host', 'Availability Zone']
        project_id = None
        if parsed_args.project:
            project_id = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
        user_id = None
        if parsed_args.user:
            user_id = identity_common.find_user(identity_client, parsed_args.user, parsed_args.user_domain).id
        all_tenants = bool(parsed_args.project) or parsed_args.all_projects
        share_type_id = None
        if parsed_args.share_type:
            share_type_id = apiutils.find_resource(share_client.share_types, parsed_args.share_type).id
        snapshot_id = None
        if parsed_args.snapshot:
            snapshot_id = apiutils.find_resource(share_client.share_snapshots, parsed_args.snapshot).id
        share_network_id = None
        if parsed_args.share_network:
            share_network_id = apiutils.find_resource(share_client.share_networks, parsed_args.share_network).id
        share_group_id = None
        if parsed_args.share_group:
            share_group_id = apiutils.find_resource(share_client.share_groups, parsed_args.share_group).id
        share_server_id = None
        if parsed_args.share_server:
            share_server_id = apiutils.find_resource(share_client.share_servers, parsed_args.share_server).id
        search_opts = {'all_tenants': all_tenants, 'is_public': parsed_args.public, 'metadata': utils.extract_key_value_options(parsed_args.property), 'extra_specs': utils.extract_key_value_options(parsed_args.extra_spec), 'limit': parsed_args.limit, 'name': parsed_args.name, 'status': parsed_args.status, 'host': parsed_args.host, 'share_server_id': share_server_id, 'share_network_id': share_network_id, 'share_type_id': share_type_id, 'snapshot_id': snapshot_id, 'share_group_id': share_group_id, 'project_id': project_id, 'user_id': user_id, 'offset': parsed_args.marker}
        if share_client.api_version >= api_versions.APIVersion('2.69'):
            search_opts['is_soft_deleted'] = parsed_args.soft_deleted
        elif getattr(parsed_args, 'soft_deleted'):
            raise exceptions.CommandError('Filtering soft deleted shares is only available with manila API version >= 2.69')
        if share_client.api_version >= api_versions.APIVersion('2.35'):
            search_opts['export_location'] = parsed_args.export_location
        elif getattr(parsed_args, 'export_location'):
            raise exceptions.CommandError('Filtering by export location is only available with manila API version >= 2.35')
        if share_client.api_version >= api_versions.APIVersion('2.36'):
            search_opts['name~'] = getattr(parsed_args, 'name~')
            search_opts['description~'] = getattr(parsed_args, 'description~')
        elif getattr(parsed_args, 'name~') or getattr(parsed_args, 'description~'):
            raise exceptions.CommandError('Pattern based filtering (name~ and description~) is only available with manila API version >= 2.36')
        data = share_client.shares.list(search_opts=search_opts)
        data = oscutils.sort_items(data, parsed_args.sort, str)
        return (column_headers, (oscutils.get_item_properties(s, columns, formatters={'Metadata': oscutils.format_dict}) for s in data))