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
class RestoreVolumeBackup(command.ShowOne):
    _description = _('Restore volume backup')

    def get_parser(self, prog_name):
        parser = super(RestoreVolumeBackup, self).get_parser(prog_name)
        parser.add_argument('backup', metavar='<backup>', help=_('Backup to restore (name or ID)'))
        parser.add_argument('volume', metavar='<volume>', nargs='?', help=_('Volume to restore to (name or ID for existing volume, name only for new volume) (default to None)'))
        parser.add_argument('--force', action='store_true', help=_('Restore the backup to an existing volume (default to False)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.sdk_connection.volume
        backup = volume_client.find_backup(parsed_args.backup, ignore_missing=False)
        volume_name = None
        volume_id = None
        try:
            volume_id = volume_client.find_volume(parsed_args.volume, ignore_missing=False).id
        except Exception:
            volume_name = parsed_args.volume
        else:
            if not parsed_args.force:
                msg = _("Volume '%s' already exists; if you want to restore the backup to it you need to specify the '--force' option")
                raise exceptions.CommandError(msg % parsed_args.volume)
        return volume_client.restore_backup(backup.id, volume_id=volume_id, name=volume_name)