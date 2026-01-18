from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from troveclient.i18n import _
class ListDatabases(command.Lister):
    _description = _('Get a list of all Databases from the instance.')
    columns = ['Name']

    def get_parser(self, prog_name):
        parser = super(ListDatabases, self).get_parser(prog_name)
        parser.add_argument(dest='instance', metavar='<instance>', help=_('ID or name of the instance.'))
        return parser

    def take_action(self, parsed_args):
        manager = self.app.client_manager.database
        databases = manager.databases
        instance = utils.find_resource(manager.instances, parsed_args.instance)
        items = databases.list(instance)
        dbs = items
        while items.next:
            items = databases.list(instance, marker=items.next)
            dbs += items
        dbs = [utils.get_item_properties(db, self.columns) for db in dbs]
        return (self.columns, dbs)