import logging
from openstackclient.identity import common as identity_common
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import constants
class DeleteResourceLock(command.Command):
    """Remove one or more resource locks."""
    _description = _('Remove one or more resource locks')

    def get_parser(self, prog_name):
        parser = super(DeleteResourceLock, self).get_parser(prog_name)
        parser.add_argument('lock', metavar='<lock>', nargs='+', help='ID(s) of the lock(s).')
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        failure_count = 0
        for lock in parsed_args.lock:
            try:
                lock = apiutils.find_resource(share_client.resource_locks, lock)
                lock.delete()
            except Exception as e:
                failure_count += 1
                LOG.error(_('Failed to delete %(lock)s: %(e)s'), {'lock': lock, 'e': e})
        if failure_count > 0:
            raise exceptions.CommandError(_('Unable to delete some or all of the specified locks.'))