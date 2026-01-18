import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
class ShowNetworkSegment(command.ShowOne):
    _description = _('Display network segment details')

    def get_parser(self, prog_name):
        parser = super(ShowNetworkSegment, self).get_parser(prog_name)
        parser.add_argument('network_segment', metavar='<network-segment>', help=_('Network segment to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_segment(parsed_args.network_segment, ignore_missing=False)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns)
        return (display_columns, data)