import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient.common._i18n import _
from manilaclient.common import constants
from manilaclient.osc import utils
class DeleteShareBackup(command.Command):
    """Delete one or more share backups."""
    _description = _('Delete one or more share backups')

    def get_parser(self, prog_name):
        parser = super(DeleteShareBackup, self).get_parser(prog_name)
        parser.add_argument('backup', metavar='<backup>', nargs='+', help=_('Name or ID of the backup(s) to delete'))
        parser.add_argument('--wait', action='store_true', default=False, help=_('Wait for share backup deletion'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        result = 0
        for backup in parsed_args.backup:
            try:
                share_backup_obj = osc_utils.find_resource(share_client.share_backups, backup)
                share_client.share_backups.delete(share_backup_obj)
                if parsed_args.wait:
                    if not osc_utils.wait_for_delete(manager=share_client.share_backups, res_id=share_backup_obj.id):
                        result += 1
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete a share backup with name or ID '%(backup)s': %(e)s"), {'backup': backup, 'e': e})
        if result > 0:
            total = len(parsed_args.backup)
            msg = _('%(result)s of %(total)s backups failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)