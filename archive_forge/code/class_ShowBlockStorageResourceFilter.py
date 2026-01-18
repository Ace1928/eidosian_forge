from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowBlockStorageResourceFilter(command.ShowOne):
    _description = _('Show filters for a block storage resource type')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('resource', metavar='<resource>', help=_('Resource to show filters for (name).'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.sdk_connection.volume
        if not sdk_utils.supports_microversion(volume_client, '3.33'):
            msg = _("--os-volume-api-version 3.33 or greater is required to support the 'block storage resource filter show' command")
            raise exceptions.CommandError(msg)
        data = volume_client.resource_filters(resource=parsed_args.resource)
        if not data:
            msg = _("No resource filter with a name of {parsed_args.resource}' exists.")
            raise exceptions.CommandError(msg)
        resource_filter = next(data)
        column_headers = ('Resource', 'Filters')
        columns = ('resource', 'filters')
        formatters = {'filters': format_columns.ListColumn}
        return (column_headers, utils.get_dict_properties(resource_filter, columns, formatters=formatters))