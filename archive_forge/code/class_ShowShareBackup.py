import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient.common._i18n import _
from manilaclient.common import constants
from manilaclient.osc import utils
class ShowShareBackup(command.ShowOne):
    """Show share backup."""
    _description = _('Show details of a backup')

    def get_parser(self, prog_name):
        parser = super(ShowShareBackup, self).get_parser(prog_name)
        parser.add_argument('backup', metavar='<backup>', help=_('ID of the share backup. '))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        backup = osc_utils.find_resource(share_client.share_backups, parsed_args.backup)
        backup._info.pop('links', None)
        return self.dict2columns(backup._info)