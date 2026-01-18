import logging
from osc_lib.cli import format_columns
from osc_lib.cli.parseractions import KeyValueAction
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc import utils as nc_osc_utils
class ShowBgpvpn(command.ShowOne):
    _description = _('Show information of a given BGP VPN')

    def get_parser(self, prog_name):
        parser = super(ShowBgpvpn, self).get_parser(prog_name)
        parser.add_argument('bgpvpn', metavar='<bgpvpn>', help=_('BGP VPN to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        id = client.find_bgpvpn(parsed_args.bgpvpn)['id']
        obj = client.get_bgpvpn(id)
        display_columns, columns = nc_osc_utils._get_columns(obj)
        data = osc_utils.get_dict_properties(obj, columns, formatters=_formatters)
        return (display_columns, data)