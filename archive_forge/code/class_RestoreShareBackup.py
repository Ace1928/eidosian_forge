import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient.common._i18n import _
from manilaclient.common import constants
from manilaclient.osc import utils
class RestoreShareBackup(command.Command):
    """Restore share backup to share"""
    _description = _('Attempt to restore share backup')

    def get_parser(self, prog_name):
        parser = super(RestoreShareBackup, self).get_parser(prog_name)
        parser.add_argument('backup', metavar='<backup>', help=_('ID of backup to restore.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_backup = osc_utils.find_resource(share_client.share_backups, parsed_args.backup)
        share_client.share_backups.restore(share_backup.id)