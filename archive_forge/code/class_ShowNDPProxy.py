import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class ShowNDPProxy(command.ShowOne):
    _description = _('Display NDP proxy details')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('ndp_proxy', metavar='<ndp-proxy>', help=_('The ID or name of the NDP proxy'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_ndp_proxy(parsed_args.ndp_proxy, ignore_missing=False)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns)
        return (display_columns, data)