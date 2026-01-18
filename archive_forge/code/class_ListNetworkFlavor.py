import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class ListNetworkFlavor(command.Lister):
    _description = _('List network flavors')

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        columns = ('id', 'name', 'is_enabled', 'service_type', 'description')
        column_headers = ('ID', 'Name', 'Enabled', 'Service Type', 'Description')
        data = client.flavors()
        return (column_headers, (utils.get_item_properties(s, columns) for s in data))