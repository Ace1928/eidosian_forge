from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from troveclient.i18n import _
class RevokeDatabaseUserAccess(command.Command):
    _description = _('Revokes access to a database for a user.')

    def get_parser(self, prog_name):
        parser = super(RevokeDatabaseUserAccess, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', help=_('ID or name of the instance.'))
        parser.add_argument('name', metavar='<name>', help=_('Name of user.'))
        parser.add_argument('--host', metavar='<host>', default=None, help=_('Optional host of user.'))
        parser.add_argument('databases', metavar='<databases>', help=_('A single database.'))
        return parser

    def take_action(self, parsed_args):
        manager = self.app.client_manager.database
        users = manager.users
        instance = utils.find_resource(manager.instances, parsed_args.instance)
        users.revoke(instance, parsed_args.name, parsed_args.databases, hostname=parsed_args.host)