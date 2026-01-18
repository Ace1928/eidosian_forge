from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from troveclient.i18n import _
class ShowDatabaseUserAccess(command.Lister):
    _description = _('Shows access details of a user of an instance.')
    columns = ['Name']

    def get_parser(self, prog_name):
        parser = super(ShowDatabaseUserAccess, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', help=_('ID or name of the instance.'))
        parser.add_argument('name', metavar='<name>', help=_('Name of user.'))
        parser.add_argument('--host', metavar='<host>', default=None, help=_('Optional host of user.'))
        return parser

    def take_action(self, parsed_args):
        manager = self.app.client_manager.database
        users = manager.users
        instance = utils.find_resource(manager.instances, parsed_args.instance)
        names = users.list_access(instance, parsed_args.name, hostname=parsed_args.host)
        access = [utils.get_item_properties(n, self.columns) for n in names]
        return (self.columns, access)