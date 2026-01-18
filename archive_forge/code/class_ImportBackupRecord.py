import logging
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
class ImportBackupRecord(command.ShowOne):
    _description = _('Import volume backup details.\n\nExported backup details contain the metadata necessary to restore to a new or\nrebuilt service instance')

    def get_parser(self, prog_name):
        parser = super(ImportBackupRecord, self).get_parser(prog_name)
        parser.add_argument('backup_service', metavar='<backup_service>', help=_('Backup service containing the backup.'))
        parser.add_argument('backup_metadata', metavar='<backup_metadata>', help=_('Encoded backup metadata from export.'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        backup_data = volume_client.backups.import_record(parsed_args.backup_service, parsed_args.backup_metadata)
        backup_data.pop('links', None)
        return zip(*sorted(backup_data.items()))