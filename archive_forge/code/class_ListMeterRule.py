import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class ListMeterRule(command.Lister):
    _description = _('List meter rules')

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        columns = ('id', 'excluded', 'direction', 'remote_ip_prefix', 'source_ip_prefix', 'destination_ip_prefix')
        column_headers = ('ID', 'Excluded', 'Direction', 'Remote IP Prefix', 'Source IP Prefix', 'Destination IP Prefix')
        data = client.metering_label_rules()
        return (column_headers, (utils.get_item_properties(s, columns) for s in data))