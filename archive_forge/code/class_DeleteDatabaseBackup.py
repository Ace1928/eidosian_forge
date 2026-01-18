from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient.i18n import _
from troveclient.osc.v1 import base
from troveclient import utils as trove_utils
class DeleteDatabaseBackup(base.TroveDeleter):
    _description = _('Deletes a backup.')

    def get_parser(self, prog_name):
        parser = super(DeleteDatabaseBackup, self).get_parser(prog_name)
        parser.add_argument('backup', nargs='+', metavar='backup', help='Id or name of backup(s).')
        return parser

    def take_action(self, parsed_args):
        db_backups = self.app.client_manager.database.backups
        self.delete_func = db_backups.delete
        self.resource = 'database backup'
        ids = []
        for backup_id in parsed_args.backup:
            if not uuidutils.is_uuid_like(backup_id):
                try:
                    backup_id = trove_utils.get_resource_id_by_name(db_backups, backup_id)
                except Exception as e:
                    msg = 'Failed to get database backup %s, error: %s' % (backup_id, str(e))
                    raise exceptions.CommandError(msg)
            ids.append(backup_id)
        self.delete_resources(ids)