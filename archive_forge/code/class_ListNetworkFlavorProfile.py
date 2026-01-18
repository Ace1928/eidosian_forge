import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class ListNetworkFlavorProfile(command.Lister):
    _description = _('List network flavor profile(s)')

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        columns = ('id', 'driver', 'is_enabled', 'meta_info', 'description')
        column_headers = ('ID', 'Driver', 'Enabled', 'Metainfo', 'Description')
        data = client.service_profiles()
        return (column_headers, (utils.get_item_properties(s, columns) for s in data))