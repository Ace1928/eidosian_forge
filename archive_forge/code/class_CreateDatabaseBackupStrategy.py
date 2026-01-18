from osc_lib.command import command
from osc_lib import utils as osc_utils
from troveclient.i18n import _
class CreateDatabaseBackupStrategy(command.ShowOne):
    _description = _('Creates backup strategy for the project or a particular instance.')

    def get_parser(self, prog_name):
        parser = super(CreateDatabaseBackupStrategy, self).get_parser(prog_name)
        parser.add_argument('--project-id', help=_('Project ID in Keystone. Only admin user is allowed to create backup strategy for other projects.'))
        parser.add_argument('--instance-id', help=_('Database instance ID.'))
        parser.add_argument('--swift-container', help=_('The container name for storing the backup data when Swift is used as backup storage backend.'))
        return parser

    def take_action(self, parsed_args):
        manager = self.app.client_manager.database.backup_strategies
        result = manager.create(instance_id=parsed_args.instance_id, swift_container=parsed_args.swift_container)
        return zip(*sorted(result.to_dict().items()))