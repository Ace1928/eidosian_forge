import logging
from openstackclient.identity import common as identity_common
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import constants
class UnsetResourceLock(command.Command):
    """Unsets a property on a resource lock."""
    _description = _('Remove resource lock properties')

    def get_parser(self, prog_name):
        parser = super(UnsetResourceLock, self).get_parser(prog_name)
        parser.add_argument('lock', metavar='<lock>', help='ID of resource lock to update.')
        parser.add_argument('--lock-reason', '--lock_reason', '--reason', dest='lock_reason', action='store_true', default=False, help='Unset the lock reason. (Default=False)')
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        if parsed_args.lock_reason:
            share_client.resource_locks.update(parsed_args.lock, lock_reason=None)