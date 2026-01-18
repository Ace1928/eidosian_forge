from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListBlockStorageResourceFilter(command.Lister):
    _description = _('List block storage resource filters')

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.sdk_connection.volume
        if not sdk_utils.supports_microversion(volume_client, '3.33'):
            msg = _("--os-volume-api-version 3.33 or greater is required to support the 'block storage resource filter list' command")
            raise exceptions.CommandError(msg)
        column_headers = ('Resource', 'Filters')
        columns = ('resource', 'filters')
        data = volume_client.resource_filters()
        formatters = {'filters': format_columns.ListColumn}
        return (column_headers, (utils.get_item_properties(s, columns, formatters=formatters) for s in data))