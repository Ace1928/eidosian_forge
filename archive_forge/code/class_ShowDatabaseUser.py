from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from troveclient.i18n import _
class ShowDatabaseUser(command.ShowOne):
    _description = _('Shows details of a database user of an instance.')

    def get_parser(self, prog_name):
        parser = super(ShowDatabaseUser, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', help=_('ID or name of the instance.'))
        parser.add_argument('name', metavar='<name>', help=_('Name of user.'))
        parser.add_argument('--host', metavar='<host>', help=_('Optional host of user.'))
        return parser

    def take_action(self, parsed_args):
        manager = self.app.client_manager.database
        db_users = manager.users
        instance = utils.find_resource(manager.instances, parsed_args.instance)
        user = db_users.get(instance, parsed_args.name, hostname=parsed_args.host)
        return zip(*sorted(user._info.items()))