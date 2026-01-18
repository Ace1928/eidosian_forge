from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from troveclient.i18n import _
class DeleteDatabaseUser(command.Command):
    _description = _('Deletes a user from an instance.')

    def get_parser(self, prog_name):
        parser = super(DeleteDatabaseUser, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', help=_('ID or name of the instance.'))
        parser.add_argument('name', metavar='<name>', help=_('Name of user.'))
        parser.add_argument('--host', metavar='<host>', help=_('Optional host of user.'))
        return parser

    def take_action(self, parsed_args):
        manager = self.app.client_manager.database
        users = manager.users
        try:
            instance = utils.find_resource(manager.instances, parsed_args.instance)
            users.delete(instance, parsed_args.name, parsed_args.host)
        except Exception as e:
            msg = _('Failed to delete user %(user)s: %(e)s') % {'user': parsed_args.name, 'e': e}
            raise exceptions.CommandError(msg)