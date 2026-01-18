import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class ListMeter(command.Lister):
    _description = _('List network meters')

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        columns = ('id', 'name', 'description', 'shared')
        column_headers = ('ID', 'Name', 'Description', 'Shared')
        data = client.metering_labels()
        return (column_headers, (utils.get_item_properties(s, columns) for s in data))