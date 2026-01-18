from osc_lib.command import command
from osc_lib import utils
from troveclient import exceptions
from troveclient.i18n import _
class ListDatabaseFlavors(command.Lister):
    _description = _('List database flavors')
    columns = ['ID', 'Name', 'RAM', 'vCPUs', 'Disk', 'Ephemeral']

    def get_parser(self, prog_name):
        parser = super(ListDatabaseFlavors, self).get_parser(prog_name)
        parser.add_argument('--datastore-type', dest='datastore_type', metavar='<datastore-type>', help=_('Type of the datastore. For eg: mysql.'))
        parser.add_argument('--datastore-version-id', dest='datastore_version_id', metavar='<datastore-version-id>', help=_('ID of the datastore version.'))
        return parser

    def take_action(self, parsed_args):
        db_flavors = self.app.client_manager.database.flavors
        if parsed_args.datastore_type and parsed_args.datastore_version_id:
            flavors = db_flavors.list_datastore_version_associated_flavors(datastore=parsed_args.datastore_type, version_id=parsed_args.datastore_version_id)
        elif not parsed_args.datastore_type and (not parsed_args.datastore_version_id):
            flavors = db_flavors.list()
        else:
            raise exceptions.MissingArgs(['datastore-type', 'datastore-version-id'])
        _flavors = []
        for f in flavors:
            if not f.id and hasattr(f, 'str_id'):
                f.id = f.str_id
            _flavors.append(utils.get_item_properties(f, self.columns))
        return (self.columns, _flavors)