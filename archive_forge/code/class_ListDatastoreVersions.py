from osc_lib.command import command
from osc_lib import utils
from troveclient import exceptions
from troveclient.i18n import _
from troveclient import utils as tc_utils
class ListDatastoreVersions(command.Lister):
    _description = _('Lists available versions for a datastore')
    columns = ['ID', 'Name', 'Version']

    def get_parser(self, prog_name):
        parser = super(ListDatastoreVersions, self).get_parser(prog_name)
        parser.add_argument('datastore', metavar='<datastore>', help=_('ID or name of the datastore'))
        return parser

    def take_action(self, parsed_args):
        datastore_version_client = self.app.client_manager.database.datastore_versions
        versions = datastore_version_client.list(parsed_args.datastore)
        ds = [utils.get_dict_properties(d.to_dict(), self.columns) for d in versions]
        return (self.columns, ds)