import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient.common._i18n import _
from manilaclient.common import constants
from manilaclient.osc import utils
class CreateShareBackup(command.ShowOne):
    """Create a share backup."""
    _description = _('Create a backup of the given share')

    def get_parser(self, prog_name):
        parser = super(CreateShareBackup, self).get_parser(prog_name)
        parser.add_argument('share', metavar='<share>', help=_('Name or ID of the share to backup.'))
        parser.add_argument('--name', metavar='<name>', default=None, help=_('Optional share backup name. (Default=None).'))
        parser.add_argument('--description', metavar='<description>', default=None, help=_('Optional share backup description. (Default=None).'))
        parser.add_argument('--backup-options', metavar='<key=value>', default={}, action=parseractions.KeyValueAction, help=_('Backup driver option key=value pairs (Optional, Default=None).'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share = osc_utils.find_resource(share_client.shares, parsed_args.share)
        body = {}
        if parsed_args.backup_options:
            body['backup_options'] = utils.extract_key_value_options(parsed_args.backup_options)
        if parsed_args.description:
            body['description'] = parsed_args.description
        if parsed_args.name:
            body['name'] = parsed_args.name
        share_backup = share_client.share_backups.create(share, **body)
        share_backup._info.pop('links', None)
        return self.dict2columns(share_backup._info)