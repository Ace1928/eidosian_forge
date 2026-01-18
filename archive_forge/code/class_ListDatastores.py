from osc_lib.command import command
from osc_lib import utils
from troveclient import exceptions
from troveclient.i18n import _
from troveclient import utils as tc_utils
class ListDatastores(command.Lister):
    _description = _('List available datastores')
    columns = ['ID', 'Name']

    def take_action(self, parsed_args):
        datastore_client = self.app.client_manager.database.datastores
        datastores = datastore_client.list()
        ds = [utils.get_item_properties(d, self.columns) for d in datastores]
        return (self.columns, ds)