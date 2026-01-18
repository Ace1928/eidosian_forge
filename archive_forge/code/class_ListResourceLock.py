import logging
from openstackclient.identity import common as identity_common
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import constants
class ListResourceLock(command.Lister):
    """Lists all resource locks."""
    _description = _('Lists all resource locks')

    def get_parser(self, prog_name):
        parser = super(ListResourceLock, self).get_parser(prog_name)
        parser.add_argument('--all-projects', action='store_true', help=_('Filter resource locks for all projects. (Admin only).'))
        parser.add_argument('--project', default=None, help=_('Filter resource locks for specific project by name or ID, combine with --all-projects (Admin only).'))
        parser.add_argument('--user', default=None, help=_('Filter resource locks for specific user by name or ID, combine with --all-projects to search across projects (Admin only).'))
        parser.add_argument('--id', metavar='<id>', default=None, help='Filter resource locks by ID. Default=None.')
        parser.add_argument('--resource', '--resource-id', '--resource_id', default=None, metavar='<resource-id>', dest='resource', help=_('Filter resource locks for a resource by ID, specify --resource-type to look up by name.'))
        parser.add_argument('--resource-type', '--resource_type', default=None, metavar='<resource_type>', help=_('Filter resource locks by type of resource.'))
        parser.add_argument('--resource-action', '--resource_action', default=None, metavar='<resource_action>', help=_('Filter resource locks by resource action.'))
        parser.add_argument('--lock-context', '--lock_context', '--context', default=None, choices=['user', 'admin', 'service'], metavar='<lock_context>', help=_('Filter resource locks by context.'))
        parser.add_argument('--since', default=None, metavar='<created_since>', help=_('Filter resource locks created since given date. The date format must be conforming to ISO8601. '))
        parser.add_argument('--before', default=None, metavar='<created_before>', help=_('Filter resource locks created before given date. The date format must be conforming to ISO8601. '))
        parser.add_argument('--limit', metavar='<limit>', type=int, default=None, help=_('Number of resource locks to list. (Default=None)'))
        parser.add_argument('--offset', metavar='<offset>', default=None, help='Starting position of resource lock records in a paginated list.')
        parser.add_argument('--sort-key', '--sort_key', metavar='<sort_key>', type=str, default=None, choices=constants.RESOURCE_LOCK_SORT_KEY_VALUES, help='Key to be sorted, available keys are %(keys)s. Default=None.' % {'keys': constants.RESOURCE_LOCK_SORT_KEY_VALUES})
        parser.add_argument('--sort-dir', '--sort_dir', metavar='<sort_dir>', type=str, default=None, choices=constants.SORT_DIR_VALUES, help='Sort direction, available values are %(values)s. OPTIONAL: Default=None.' % {'values': constants.SORT_DIR_VALUES})
        parser.add_argument('--detailed', dest='detailed', metavar='<0|1>', nargs='?', type=int, const=1, default=0, help='Show detailed information about filtered resource locks.')
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        columns = LOCK_SUMMARY_ATTRIBUTES if not parsed_args.detailed else LOCK_DETAIL_ATTRIBUTES
        project_id = None
        user_id = None
        if parsed_args.project:
            project_id = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
        if parsed_args.user:
            user_id = identity_common.find_user(identity_client, parsed_args.user, parsed_args.user_domain).id
        all_projects = bool(parsed_args.project) or parsed_args.all_projects
        resource_id = parsed_args.resource
        resource_type = parsed_args.resource_type
        if resource_type is not None:
            if resource_type not in RESOURCE_TYPE_MANAGERS:
                raise exceptions.CommandError(_('Unsupported resource type'))
            if resource_id is not None:
                res_manager = RESOURCE_TYPE_MANAGERS[resource_type]
                resource_id = osc_utils.find_resource(getattr(share_client, res_manager), parsed_args.resource).id
        elif resource_id and (not uuidutils.is_uuid_like(resource_id)):
            raise exceptions.CommandError(_('Provide resource ID or specify --resource-type.'))
        search_opts = {'all_projects': all_projects, 'project_id': project_id, 'user_id': user_id, 'id': parsed_args.id, 'resource_id': resource_id, 'resource_type': parsed_args.resource_type, 'resource_action': parsed_args.resource_action, 'lock_context': parsed_args.lock_context, 'created_before': parsed_args.before, 'created_since': parsed_args.since, 'limit': parsed_args.limit, 'offset': parsed_args.offset}
        resource_locks = share_client.resource_locks.list(search_opts=search_opts, sort_key=parsed_args.sort_key, sort_dir=parsed_args.sort_dir)
        return (columns, (osc_utils.get_item_properties(m, columns) for m in resource_locks))