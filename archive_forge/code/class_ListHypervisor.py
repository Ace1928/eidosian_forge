import json
import re
from novaclient import exceptions as nova_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
class ListHypervisor(command.Lister):
    _description = _('List hypervisors')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--matching', metavar='<hostname>', help=_('Filter hypervisors using <hostname> substringHypervisor Type and Host IP are not returned when using microversion 2.52 or lower'))
        pagination.add_marker_pagination_option_to_parser(parser)
        parser.add_argument('--long', action='store_true', help=_('List additional fields in output'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        list_opts = {}
        if parsed_args.matching and (parsed_args.marker or parsed_args.limit):
            msg = _('--matching is not compatible with --marker or --limit')
            raise exceptions.CommandError(msg)
        if parsed_args.marker:
            if not sdk_utils.supports_microversion(compute_client, '2.33'):
                msg = _('--os-compute-api-version 2.33 or greater is required to support the --marker option')
                raise exceptions.CommandError(msg)
            list_opts['marker'] = parsed_args.marker
        if parsed_args.limit:
            if not sdk_utils.supports_microversion(compute_client, '2.33'):
                msg = _('--os-compute-api-version 2.33 or greater is required to support the --limit option')
                raise exceptions.CommandError(msg)
            list_opts['limit'] = parsed_args.limit
        if parsed_args.matching:
            list_opts['hypervisor_hostname_pattern'] = parsed_args.matching
        column_headers = ('ID', 'Hypervisor Hostname', 'Hypervisor Type', 'Host IP', 'State')
        columns = ('id', 'name', 'hypervisor_type', 'host_ip', 'state')
        if parsed_args.long:
            if not sdk_utils.supports_microversion(compute_client, '2.88'):
                column_headers += ('vCPUs Used', 'vCPUs', 'Memory MB Used', 'Memory MB')
                columns += ('vcpus_used', 'vcpus', 'memory_used', 'memory_size')
        data = compute_client.hypervisors(**list_opts, details=True)
        return (column_headers, (utils.get_item_properties(s, columns) for s in data))