import itertools
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class ShowNetworkSegmentRange(command.ShowOne):
    _description = _('Display network segment range details')

    def get_parser(self, prog_name):
        parser = super(ShowNetworkSegmentRange, self).get_parser(prog_name)
        parser.add_argument('network_segment_range', metavar='<network-segment-range>', help=_('Network segment range to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        network_client = self.app.client_manager.network
        try:
            network_client.find_extension('network-segment-range', ignore_missing=False)
        except Exception as e:
            msg = _('Network segment range show not supported by Network API: %(e)s') % {'e': e}
            raise exceptions.CommandError(msg)
        obj = network_client.find_network_segment_range(parsed_args.network_segment_range, ignore_missing=False)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns)
        data = _update_additional_fields_from_props(columns, props=data)
        return (display_columns, data)