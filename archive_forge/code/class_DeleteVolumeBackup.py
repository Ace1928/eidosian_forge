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
class DeleteVolumeBackup(command.Command):
    _description = _('Delete volume backup(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteVolumeBackup, self).get_parser(prog_name)
        parser.add_argument('backups', metavar='<backup>', nargs='+', help=_('Backup(s) to delete (name or ID)'))
        parser.add_argument('--force', action='store_true', default=False, help=_('Allow delete in state other than error or available'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.sdk_connection.volume
        result = 0
        for backup in parsed_args.backups:
            try:
                backup_id = volume_client.find_backup(backup, ignore_missing=False).id
                volume_client.delete_backup(backup_id, ignore_missing=False, force=parsed_args.force)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete backup with name or ID '%(backup)s': %(e)s") % {'backup': backup, 'e': e})
        if result > 0:
            total = len(parsed_args.backups)
            msg = _('%(result)s of %(total)s backups failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)