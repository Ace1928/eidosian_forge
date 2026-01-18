import itertools
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class ListNetworkSegmentRange(command.Lister):
    _description = _('List network segment ranges')

    def get_parser(self, prog_name):
        parser = super(ListNetworkSegmentRange, self).get_parser(prog_name)
        parser.add_argument('--long', action='store_true', help=_('List additional fields in output'))
        used_group = parser.add_mutually_exclusive_group()
        used_group.add_argument('--used', action='store_true', help=_('List network segment ranges that have segments in use'))
        used_group.add_argument('--unused', action='store_true', help=_('List network segment ranges that have segments not in use'))
        available_group = parser.add_mutually_exclusive_group()
        available_group.add_argument('--available', action='store_true', help=_('List network segment ranges that have available segments'))
        available_group.add_argument('--unavailable', action='store_true', help=_('List network segment ranges without available segments'))
        return parser

    def take_action(self, parsed_args):
        network_client = self.app.client_manager.network
        try:
            network_client.find_extension('network-segment-range', ignore_missing=False)
        except Exception as e:
            msg = _('Network segment ranges list not supported by Network API: %(e)s') % {'e': e}
            raise exceptions.CommandError(msg)
        filters = {}
        data = network_client.network_segment_ranges(**filters)
        headers = ('ID', 'Name', 'Default', 'Shared', 'Project ID', 'Network Type', 'Physical Network', 'Minimum ID', 'Maximum ID')
        columns = ('id', 'name', 'default', 'shared', 'project_id', 'network_type', 'physical_network', 'minimum', 'maximum')
        if parsed_args.available or parsed_args.unavailable or parsed_args.used or parsed_args.unused:
            parsed_args.long = True
        if parsed_args.long:
            headers = headers + ('Used', 'Available')
            columns = columns + ('used', 'available')
        display_props = tuple()
        for s in data:
            props = utils.get_item_properties(s, columns)
            if parsed_args.available and _is_prop_empty(columns, props, 'available') or (parsed_args.unavailable and (not _is_prop_empty(columns, props, 'available'))) or (parsed_args.used and _is_prop_empty(columns, props, 'used')) or (parsed_args.unused and (not _is_prop_empty(columns, props, 'used'))):
                continue
            if parsed_args.long:
                props = _update_additional_fields_from_props(columns, props)
            display_props += (props,)
        return (headers, display_props)