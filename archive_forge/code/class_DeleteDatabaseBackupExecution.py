from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient.i18n import _
from troveclient.osc.v1 import base
from troveclient import utils as trove_utils
class DeleteDatabaseBackupExecution(command.Command):
    _description = _('Deletes an execution.')

    def get_parser(self, prog_name):
        parser = super(DeleteDatabaseBackupExecution, self).get_parser(prog_name)
        parser.add_argument('execution', metavar='<execution>', help=_('ID of the execution to delete.'))
        return parser

    def take_action(self, parsed_args):
        database_backups = self.app.client_manager.database.backups
        database_backups.execution_delete(parsed_args.execution)