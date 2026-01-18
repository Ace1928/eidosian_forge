import logging
from openstackclient.identity import common as identity_common
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import constants
class CreateResourceLock(command.ShowOne):
    """Create a new resource lock."""
    _description = _('Lock a resource action from occurring on a resource')

    def get_parser(self, prog_name):
        parser = super(CreateResourceLock, self).get_parser(prog_name)
        parser.add_argument('resource', metavar='<resource_name_or_id>', help='Name or ID of resource to lock.')
        parser.add_argument('resource_type', metavar='<resource_type>', help='Type of the resource (e.g.: share, access).')
        parser.add_argument('--resource-action', '--resource_action', metavar='<resource_action>', default='delete', help='Action to lock on the resource (default="delete")')
        parser.add_argument('--lock-reason', '--lock_reason', '--reason', metavar='<lock_reason>', help='Reason for the resource lock.')
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        resource_type = parsed_args.resource_type
        if resource_type not in RESOURCE_TYPE_MANAGERS:
            raise exceptions.CommandError(_('Unsupported resource type'))
        res_manager = RESOURCE_TYPE_MANAGERS[resource_type]
        resource = osc_utils.find_resource(getattr(share_client, res_manager), parsed_args.resource)
        resource_lock = share_client.resource_locks.create(resource.id, resource_type, parsed_args.resource_action, parsed_args.lock_reason)
        resource_lock._info.pop('links', None)
        return self.dict2columns(resource_lock._info)