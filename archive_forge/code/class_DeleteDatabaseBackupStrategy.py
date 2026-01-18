from osc_lib.command import command
from osc_lib import utils as osc_utils
from troveclient.i18n import _
class DeleteDatabaseBackupStrategy(command.Command):
    _description = _('Deletes backup strategy.')

    def get_parser(self, prog_name):
        parser = super(DeleteDatabaseBackupStrategy, self).get_parser(prog_name)
        parser.add_argument('--project-id', help=_('Project ID in Keystone. Only admin user is allowed to delete backup strategy for other projects.'))
        parser.add_argument('--instance-id', help=_('Database instance ID.'))
        return parser

    def take_action(self, parsed_args):
        manager = self.app.client_manager.database.backup_strategies
        manager.delete(instance_id=parsed_args.instance_id, project_id=parsed_args.project_id)